# TODAY’S WORK, TOMORROW’S WORKFORCE: INCENTIVIZING LEARNING IN AI-ASSISTED WORK

Moritz Diener<sup>∗</sup> Karlsruhe Institute of Technology (KIT) Karlsruhe, Germany moritz.diener@kit.edu

Philipp Spitzer Karlsruhe Institute of Technology (KIT) Karlsruhe, Germany philipp.spitzer@kit.edu

Joshua Holstein Karlsruhe Institute of Technology (KIT) Karlsruhe, Germany joshua.holstein@kit.edu

Daniel Hendriks<sup>∗</sup> Karlsruhe Institute of Technology (KIT) Karlsruhe, Germany daniel.hendriks@kit.edu

Gerhard Satzger Karlsruhe Institute of Technology (KIT) Karlsruhe, Germany gerhard.satzger@kit.edu

Wei Sun  
IBM Research  
USA  
sunw@us.ibm.com

## ABSTRACT

At work, people develop skills through experience: novices acquire skills by completing tasks, while experts retain theirs through continued practice. AI assistance changes this process by offering workers a choice between solving tasks independently and accepting the AI’s answers. While relying on AI saves immediate effort, it can also displace the practice needed to acquire and retain skills, including those required to oversee the AI. We extend an existing accept-or-solve model to a sequence of tasks in which workers’ skills evolve through independent practice and reliance on AI. We compare myopic workers, who maximize immediate utility, with strategic workers, who also value their future skills. We show that novices can become trapped at low skill levels, while experts may lose skill, even as AI-assisted outcome accuracy remains high. Strategic workers, by contrast, may choose to solve independently even when AI is more accurate, recognizing the future value of acquiring and retaining their skills. We then study incentives that reward correct independent work and characterize the payments needed to encourage skill acquisition and retention. We show that independent work can become self-sustaining once workers acquire sufficient skill, eliminating the need for continued incentives. Counterintuitively, supporting a higher skill target can sometimes reduce long-run incentive costs, either by enabling self-sustaining independent work or by lowering the payments needed to maintain proficiency. Our findings thereby shift the focus from whether workers should rely on AI on individual tasks to how organizations can sustain human expertise over time, and which skills are worth investing in.

Keywords human–AI collaboration · mechanism design · incentive design · reliance · future of work

## 1 Introduction

Astronauts on a six-month mission aboard the International Space Station need little physical strengthfor their work, but without gravity and physical effort, muscles and bones deteriorate. Since no work in space requires this physical strength, this loss often goes unnoticed until astronauts return to Earth. Once this loss becomes evident, space agencies began scheduling about two and a halfhours ofexercise into every mission day. This exercise retains the capacity that astronauts need once gravity returns. What gravity and physical effort are for an astronaut’s body is cognitive effort for a worker’s skills.

Traditionally, work has trained the people who perform it: novices acquire skill by solving tasks, and experts retain their skill through regular use [11, 25, 41]. AI loosens this link as workers can now accept an AI’s answer instead of solving a task themselves [19], often finishing faster and with higher quality [23]. The effort that AI saves, however, is the very effort through which workers learn. For example, novices who relied on AI while learning a new programming library acquired less skill [52], and endoscopists who had worked with AI detected fewer adenomas once the AI was withdrawn [17]. Work performed with AI assistance therefore risks workers not acquiring or retaining skill.

Even as AI capabilities grow, acquiring and retaining skill remains important because skills build on each other and are therefore needed to acquire other skills [28]. For example, programmers may use AI for routine coding yet still need coding skill to acquire additional skills that enable them to debug failures and make architectural judgments. This matters most early in a career, when novices perform routine tasks to build the foundation for becoming an expert [13, 34]. However, the importance of skills also extends to a longer horizon, where it is needed to oversee the AI itself, i.e., to detect its errors and take over when it fails [6, 38]. For high-risk systems, oversight is even legally mandatory through the EU AI Act [26].

Research on human–AI decision-making has evaluated workers’ choice to solve alone or use an AI by appropriate reliance: a worker should accept the AI’s answer when it is more likely to be correct and solve the task independently otherwise [42, 51]. This accept-or-solve choice is represented in formal models by maximizing the utility of a worker on a single task [7, 33]. In particular, Holstein et al. [33] show that independent effort can make accepting the AI’s answer rational even when the worker is more skilled and therefore more likely to answer correctly. With skill held fixed, however, these models miss two dynamics: A novice acquires skill by solving tasks that the AI would answer more accurately, which appropriate reliance scores as underreliance. Similarly, an expert who accepts the AI’s answer to save effort gradually loses their advantage over the AI. Dynamic models relax the assumption of fixed skill: Huang and Vishnoi [36] show that reliance driven by observed performance can lock people into low skill. Closest to our work, Aouad et al. [3] model dynamic skill and show that AI assistance can make workers less productive by saving the effort that leads to skill acquisition. Their workers, however, choose how much effort to exert rather than whether to accept the AI’s answer, so it remains open how the choice that reliance research evaluates changes once skill evolves. Once skill evolves, the answer depends on whether workers weigh only the current task or also how today’s choice shapes the skill they bring to future tasks. This motivates our first research question:

RQ1: When do workers solve tasks independently instead of accepting an AI’s answer, depending on their skill, the AI’s accuracy, and their strategic behavior?

Valuing future skill, however, does not ensure that workers acquire or retain it. Practice pays off for a worker only if the skill gained earns more on future tasks than the effort and accuracy given up today. Some skills an organization needs, such as overseeing the AI, may not raise the worker’s own payoff at all. Organizations have long paid for such practice implicitly, because the tasks they assigned trained workers as a by-product [1, 10, 11]. Once AI can complete these tasks, organizations must decide explicitly which independent work to keep, much as space agencies schedule exercise rather than leaving it to the crew. One lever is to pay workers for correct independent answers. Holstein et al. [33] propose such an incentive to offset the effort of solving, which restores appropriate reliance. It does not, however, induce learning. A worker who is less accurate than the AI gives up correct answers as well as effort when solving, so a payment that covers only effort leaves them accepting and never practicing. This leads to our second research question:

RQ2: How can incentives encourage workers to solve tasks independently and thereby acquire and maintain their skills?

To answer these questions, we extend the accept-or-solve model of Bansal et al. [7] and Holstein et al. [33] from a single task to a sequence of tasks. We adopt the skill model of Aouad et al. [3], in which skill improves through independent work and decays at a constant rate, but replace their choice of how much effort to exert with the accept-or-solve choice. A myopic worker weighs only the effort and accuracy of the current task, whereas a strategic worker also weighs today’s effort against future skill. We then examine how incentives for correct independent answers can encourage workers to acquire and retain skill. Our work makes three contributions:

• Dynamic accept-or-solve model. We extend the accept-or-solve framework with skill acquisition and decay and characterize how the value of future skill shapes reliance decisions. Strategic workers may solve independently even when the AI is more accurate, because doing so builds or preserves their skill.

• Low-skill trap and expert deskilling. We identify conditions under which reliance on AI leads to stagnating or eroding skill without a visible decline in task performance. Novices remain trapped at low skill levels, and experts lose proficiency while output accuracy stays at the AI’s level.

• Learning incentives. We derive the minimum payment for correct independent answers that makes myopic workers acquire and retain skill. A higher skill target can reduce long-run incentive costs, either by making independent work self-sustaining or by lowering the payment needed to retain skill.

Together, these findings motivate an apprenticeship perspective on AI-assisted work, in which organizations pay for the practice that AI-assisted work no longer provides.

## 2 Related Work

Novices learn by taking part in actual work, starting with simple tasks and gradually taking on more central ones [41]. Experts support novices by demonstrating how tasks are done, giving feedback while they practice, and reducing their assistance as they become more skilled [20]. Organizations finance such practice when labor-market frictions allow them to capture part of its payoff [1, 10], and apprentices contribute through low wages and routine work that is valuable to their employers [27, 30]. Automation and AI weaken this link, as they restrict the effort through which workers learn [9], limit the skill that novices develop when relying on tools they do not understand [2], change which skill is valued [5], and take over the entry-level work through which novices used to learn [56].

Even where workers still perform the tasks themselves, AI assistance can lower their unaided skill: students who used AI without restriction performed worse once assistance was removed [8], as did learners who requested AI help during practice [55]. Professionals likewise lost skills after automation took over their tasks [17, 46, 48]. These losses are consistent with shifting mental effort to external aids [49] and with skill decay through disuse [4], and echo the longstanding concern that automation causes skill decay needed to oversee it [6]. Whether assistance harms learning depends on its design and use: a guided AI tutor avoided the harm of unrestricted access [8], and contrastive explanations improved decision-making skills [15]. Greater independent problem-solving effort predicted greater skill acquisition [55], consistent with evidence that effortful practice and initial failure improve learning [12, 39].

Whereas these studies examine how AI assistance affects learning, research on appropriate reliance asks when people should accept the AI’s answer on a given task, based on human and AI accuracy [32, 42, 51], and how cognitive forcing functions can reduce overreliance on AI advice [14]. Some studies in this line also measure learning alongside accuracy [15, 29, 50] or tailor AI assistance to support learning [16]. They vary how the AI assists, but do not model how workers trade off today’s accuracy against future skill. Formal models capture the worker’s choice between accepting the AI’s answer and solving independently. Bansal et al. [7] formalize this choice as utility maximization, which Holstein et al. [33] extend with effort costs. They show that accepting can then be rational even when the worker is more accurate, and derive an incentive for correct independent answers that offsets this effort-induced overreliance. Both models, however, evaluate a single task with fixed skill.

Dynamic models relax this assumption of fixed skill. In the model of Huang and Vishnoi [36], workers acquire skill through practice, lose it through disuse, and rely on AI based on its observed performance. As AI becomes more capable, such reliance locks people into low skill from a wider range of starting levels. Closest to our work, Aouad et al. [3] model skill as a birth–death process with effort-dependent learning and exogenous decay, and show that AI assistance can undermine long-run skill and productivity by reducing effort, even among strategic workers. We differ from Aouad et al. [3] in three ways. First, both learning and decay depend on the worker’s choice on each task, so solving can let the worker acquire skill, whereas accepting the AI’s answer can let skill decay. Second, because we retain the accept-or-solve model of Bansal et al. [7], Holstein et al. [33], we can compare dynamic choices with appropriate reliance. Third, whereas apprenticeship models assume that routine work pays for practice [27], we derive the minimum incentive an organization must pay for solving independently to be rational, distinguishing temporary support to acquire skill from recurring support to retain it. We thereby extend to human–AI collaboration the insight of fairness research that single-step evaluations miss how decisions shape people over time [22, 43].

## 3 A Dynamic Accept-or-Solve Model

We extend the existing accept-or-solve model of Bansal et al. [7] from a single task to a sequence of tasks, where a worker’s skills vary based on their previous decisions. In the following, we describe how we model both skill changes and the utility of a single task.

## 3.1 Skill Dynamics

Consider a sequence of tasks $t = 0 , 1 , 2 , . . .$ . from one taskfamily, such as reviewing code changes for bugs. Let $p _ { t }$ denote the worker’s unaided accuracy, the probability of answering task t correctly without AI assistance. We represent skill using a finite set of ordered levels:

$$
\mathcal { P } = \{ a _ { 1 } < a _ { 2 } < \cdots < a _ { N } \} \subset ( 0 , 1 ] .\tag{1}
$$

For any skill level $p = a _ { i } \in \mathcal { P }$ , let $p ^ { + } = a _ { i + 1 }$ and $p ^ { - } = a _ { i - 1 }$ denote the adjacent higher and lower levels, respectively, with $p ^ { \dagger } = p$ at the highest level and $p ^ { - } = p$ at the lowest.

When accepting AI, the worker submits the AI’s answer without spending effort. This answer is correct with probability $P _ { A I }$ , but skill goes unused and decays by one level to $p ^ { - }$ with probability $\phi _ { d } \in ( 0 , 1 ] ;$ ; otherwise, it stays at p. When solving independently, the worker instead spends effort $\lambda > 0$ and submits their own answer, which is correct with probability $p .$ In return for this effort, the worker acquires skill and rises one level to $p ^ { + }$ with probability $\phi _ { u } \in ( 0 , 1 ]$ otherwise, skill stays at $p .$ Table 1 summarizes both choices.

Learning depends on the effort of practice rather than on a correct answer, so even an incorrect independent solve can lead to skill acquisition [21, 39]. In line with Corbett and Anderson [21], we assume $\phi _ { u }$ and $\phi _ { d }$ to be constant across skill levels within a task family. As in Bansal et al. [7], the worker knows their own skill level $p$ and the AI’s accuracy $P _ { A I }$ . Appendix A.1 motivates this Markov chain and visualizes its dynamics.

Formally, solving instead of accepting thus raises the worker’s expected skill level on the next task by

$$
\mathbb { E } [ p _ { t + 1 } \ | \ p _ { t } = p , \mathrm { S o l v e } ] - \mathbb { E } [ p _ { t + 1 } \ | \ p _ { t } = p , \mathrm { A c c e p t } ] = \phi _ { u } ( p ^ { + } - p ) + \phi _ { d } ( p - p ^ { - } ) \geq 0 .\tag{2}
$$

The first term is the skill the worker can acquire, and the second is the skill loss that solving avoids. Appendix A.1 derives Eq. (2) and the expected number of tasks until skill rises by one level.

## 3.2 Single-Task Utility and Running Example

As in Bansal et al. [7], the worker’s utility depends on the choice and on whether the final answer is correct (see Table 1). A correct final answer earns $\gamma > 0$ , an incorrect one earns zero, and solving oneself requires the effort cost λ. While Bansal et al. [7] reward a correct answer with $\gamma ,$ they also penalize an incorrect one with β. Let $Y _ { t } \in \{ 0 , 1 \}$ indicate whether the final answer on task t is correct, and let $U _ { t }$ denote the worker’s realized utilityfrom that task. At current skill $p ,$ the expected utilities of accepting and of solving are

$$
u _ { A } = \gamma P _ { A I } , \qquad u _ { S } ( p ) = \gamma p - \lambda .\tag{3}
$$

The immediate payoff gap therefore is the difference between the expected payoff from accepting AI and the expected payoff from solving independently:

$$
\Delta u ( p ) = u _ { A } - u _ { S } ( p ) = \underbrace { \lambda } _ { \mathrm { e f f o r t ~ c o s t } } + \underbrace { \gamma ( P _ { A I } - p ) } _ { \mathrm { a c c u r a c y ~ g a p } } .\tag{4}
$$

A positive gap means accepting the AI’s answer immediately pays off compared to solving independently, whereas a negative gap means solving oneself pays off more.

Each choice of a worker affects both today’s payoff and tomorrow’s skill. For each task, the worker can either accept the AI’s answer or solve the task independently. Accepting saves effort and yields a higher payoff today, unless the worker’s accuracy advantage over the AI outweighs the effort cost. However, only by solving independently can the worker acquire skill and prevent its decay. Each choice therefore weighs today’s payoff against the worker’s future skill.

Table 1: One task in the dynamic accept-or-solve model. Utilities by choice and correctness of the final answer follow Bansal et al. [7, Table 1]; the last column is this paper’s extension. Skill otherwise stays at $p ,$ with $p ^ { - } = p$ at the lowest and $p ^ { + } = p$ at the highest level.
<table><tr><td rowspan="2">Choice</td><td rowspan="2">Correct with prob.</td><td colspan="2">Utility if final answer is</td><td rowspan="2">Expected utility</td><td rowspan="2">Skill for the next task</td></tr><tr><td>correct</td><td>incorrect</td></tr><tr><td>Accept AI</td><td> $P _ { A I }$ </td><td> $\gamma$ </td><td>0</td><td> $u _ { A } = \gamma P _ { A I }$ </td><td>falls to  $p ^ { - }$  with probability  $\phi _ { d }$ </td></tr><tr><td>Solve independently</td><td> $p$ </td><td> $\gamma - \lambda$ </td><td> $- \lambda$ </td><td> $u _ { S } ( p ) = \gamma p - \lambda$ </td><td>rises to  $p ^ { + }$  with probability  $\phi _ { u }$ </td></tr></table>

Running example. To illustrate the model, imagine a software developer who reviews code changesfor bugs with an AI assistant. Without help, Anna, a novice,finds the bug with 40% probability; Sam, an expert,finds the bug with 80% probability. Assume that skill ranges over $\dot { \mathcal { P } } = \{ . 4 0 , . 5 \dot { 0 } , . 6 0 , . 7 0 , . \dot { 8 } 0 \}$ . With $\phi _ { u } = . 0 2$ and $\phi _ { d } = . 0 1$ , a developer rises one level after $1 / \phi _ { u } = 5 0$ solved tasks on average andfalls one level after $1 / \phi _ { d } = 1 0 0$ accepted tasks. Each bugfound earns $\gamma = 1 _ { : }$ , and reviewing a change without the AI costs effort $\lambda = . 0 5$ . Anna and Sam work with one oftwo AIs. The competent AI $( P _ { A I } = . 6 2 )$ is more accurate than Anna, but Sam’s accuracy advantage over it (.18) exceeds the effort cost. Sam is also more accurate than the near-expert AI $( P _ { A I } = . 7 7 )$ , but his advantage (.03)falls short ofthe effort cost. Appendix B shows that our illustrative results hold across a range ofparameter values.

## 4 Myopic and Strategic Workers

To answer our first research question, we compare two workers who differ only in how strategic they are when solving tasks independently. A myopic worker maximizes the utility of the current task, whereas a strategic worker maximizes the discounted sum of utility over current and future tasks, and therefore values the skill that today’s choice builds or preserves. For each worker, we ask how appropriate reliance would evaluate their choices.

## 4.1 The Myopic Worker

Research on human–AI decision-making mostly assesses reliance by the accuracy of the current answer alone [32, 51]. We apply appropriate reliance at the level of average accuracy, i.e., we compare the worker’s skill level, which indicates the probability of solving correctly, with the AI’s accuracy. Under appropriate reliance, a worker solves if $p \geq P _ { A I }$ and accepts otherwise. Accepting when $p > P _ { A I }$ is overreliance, and solving when $p < P _ { A I }$ is underreliance.

A myopic worker also weighs the effort of solving. Maximizing only today’s expected utility, the worker solves if $\Delta u ( p ) \leq 0$ , or equivalently

$$
p \ge P _ { A I } + \frac { \lambda } { \gamma } .\tag{5}
$$

The myopic threshold exceeds that of appropriate reliance by $\lambda / \gamma$ . An expert who beats the AI by less than this margin therefore still accepts, which is the effort-induced overreliance described by Holstein et al. [33]. Yet, neither appropriate reliance nor the myopic worker considers that the same choice also shapes tomorrow’s skill (2).

Running example. With the competent AI, the myopic threshold is $. 6 2 + . 0 5 = . 6 7 .$ . Anna (.40) therefore accepts, and only developers at .70 or above solve. With the near-expert AI, the threshold rises to .82, above even Sam. Although Sam (.80) is more accurate than the AI, Samfaces $\Delta u \overset { - } { = } . 0 5 + ( . 7 7 - . 8 0 ) = . 0 2 > 0$ and accepts.

## 4.2 The Strategic Worker

A strategic worker recognizes that solving a task today can raise or preserve their skill and thereby their utility on future tasks. To weigh today’s utility against future utility, the worker discounts the utility of the task t steps ahead by $\delta ^ { t }$ , with a discount factor $\delta \in [ 0 , 1 \rangle$ ). This discount factor is the probability that the worker expects to perform another task of the same family after the current one, and it does not depend on whether the worker accepts or solves. The weight $\delta ^ { t }$ is then the probability that the worker still performs these tasks t steps ahead, and the worker expects to perform $H = 1 / ( 1 - \bar { \delta } )$ tasks, including the current one (Appendix A.4). Although the sequence of tasks itself does not end, the worker may expect to leave it, for example by moving to another role. A myopic worker has $\delta = 0$ and thus $H = 1$ whereas a strategic worker has $\delta > 0$ and thus $H > 1$

Let a policy π specify the worker’s accept-or-solve choice at each skill level, and define the lifetime value $V ( p )$ as the greatest expected discounted utility starting from skill p:

$$
V ( p ) = \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \delta ^ { t } U _ { t } \Bigg | p _ { 0 } = p \right] .\tag{6}
$$

By the principle of optimality, the worker chooses on each task whichever option yields the higher sum of current utility and discounted lifetime value from the next task on. This choice satisfies the Bellman equation

$$
\begin{array} { c } { { V ( p ) = \displaystyle \operatorname* { m a x } \left\{ u _ { A } + \delta [ ( 1 - \phi _ { d } ) V ( p ) + \phi _ { d } V ( p ^ { - } ) ] , \right. } } \\ { { \left. u _ { S } ( p ) + \delta [ ( 1 - \phi _ { u } ) V ( p ) + \phi _ { u } V ( p ^ { + } ) ] \right\} . } } \end{array}\tag{7}
$$

![](images/97758147717ad8794675dc5ed67a75a05e1990f5c2ae1102ebe7f29ed2a9b402.jpg)  
Figure 1: Rational choices for a strategic worker at AI accuracy $P _ { A I }$ and expected horizon H. Blue shading indicates solving independently, and red shading indicates accepting the AI’s answer. The braces show the appropriate-reliance benchmark, which favors solving when $p _ { 0 } \geq P _ { A I }$ . Hatching marks where the strategic choice differs from that benchmark. Points on the dashed line show the running examples at $H = 1 0 0 . \left( \mathbf { A } \right)$ The novice $( p _ { 0 } = . 5 0 )$ accepts the competent AI at $H = 1 0 0$ despite the opportunity to build skill. (B) The expert $( p _ { 0 } = . 8 0 )$ accepts the near-expert AI at every horizon. Appendix A.7 derives the strategic boundaries.

The first expression is the value of accepting AI now, including possible skill decay. The second is the value of solving now, including possible learning. Comparing the two expressions gives the strategic worker’s decision rule:

$$
\begin{array} { r } { \underbrace { \Delta u ( p ) } _ { \mathrm { i m m e d i a t e \ p a y o f f g a p } } \le \underbrace { \delta \left[ \phi _ { u } \left( V ( p ^ { + } ) - V ( p ) \right) + \phi _ { d } \left( V ( p ) - V ( p ^ { - } ) \right) \right] } _ { \mathrm { f u u r e \ v a l u e \ o f \ s o l v i n g } } . } \end{array}\tag{8}
$$

The right-hand side captures the future payoff of possible learning and of avoided skill decay. When $\delta = 0 _ { : }$ , it vanishes, and Eq. (8) reduces to the myopic rule in Eq. (5). Appendix A.5 derives Eq. (7), and Appendix A.6 explains how we solve it exactly on the finite skill grid.

Because of this future payoff, a strategic worker may solve independently even when the AI is more accurate $( p < P _ { A I } )$ Figure 1 compares these choices with appropriate reliance in the running example. Whether a worker starts to solve depends on the horizon and the learning probability only through $\phi _ { u } ( H - 1 )$ , the number of skill levels the worker can expect to acquire over the remaining tasks. The decay probability plays no role in this decision, as a worker who keeps solving never loses skill (Appendix B.2).

Running example. Let $\delta = . 9 9$ , so Anna and Sam each expect 100 remaining tasks. At .60 with the competent AI, Anna faces $\Delta u = . 0 7$ and still solves, whereas a myopic Anna accepts and risksfalling back to .50. Solvingfirst costs Anna expected utility, a shortfall that peaks at 1.30 after 38 tasks and is recouped after 80 tasks. Over all remaining tasks, the discounted gain relative to always accepting is 4.06 (Appendix A.10).

Strategic behavior can make solving worthwhile even when the AI is more accurate. A myopic worker maximizes the payoff of the current task and solves only if their accuracy advantage over the AI outweighs the effort cost. Such a worker may therefore accept even when the AI is less accurate. A strategic worker also values skill for future tasks and may solve even when the AI is more accurate because the acquired skill may increase future payoffs.

## 4.3 The Low-Skill Trap and Expert Deskilling

Valuing future skill, however, does not guarantee that a worker will acquire or retain skill. Even for a strategic worker, acquiring skill may cost more than its future payoff justifies—a situation we call the low-skill trap. Retaining existing skill may likewise offer no payoff advantage over accepting AI, a situation we term expert deskilling.

The low-skill trap arises when learning does not pay off. To acquire skill a novice must solve independently, forgoing immediate payoff from accepting the AI’s answer, until reaching a skill level at which solving pays on its own. The lifetime value in Eq. (8) accounts for the entire horizon H with their opportunities for acquiring skill: the worker weighs the benefits of potentially becoming an expert. However, at a low skill level and a short horizon, the discounted utility can be smaller than the accumulated cost of reaching a skill level through the effort of solving independently. Put simply, solving independently then does not pay off – the worker chooses to accept the AI’s answers and skill can only retain or decay, leaving the worker trapped at a low skill level. Appendix A.8 states the condition under which the trap arises.

Expert deskilling arises when retaining existing skill offers no payoff advantage over accepting AI. At the highest skill level, further learning is impossible, so only the retention term in Eq. (8) remains. That retention of skill is only valuable to the worker if it improves future payoffs. Suppose $a _ { N } < P _ { A I } + \lambda / \gamma$ . Then even experts at the highest skill level earn less in expectation by solving, $\gamma a _ { N } - \lambda .$ , than by accepting, $\gamma P _ { A I } .$ , and every lower skill level earns less still. Accepting on every task is then rational regardless of starting skill, so $\dot { V } ( p ) = \gamma P _ { A I } \dot { / } ( 1 - \delta )$ for every $p \in \mathcal P$ . Both future-value terms are zero, and strategic behavior cannot make retention worthwhile.

The low-skill trap and expert deskilling hold across all examined parameter values (Appendix B). For every finite horizon, there is a range of AI accuracies in which independent work would pay at the top level, yet a novice accepts. Longer horizons and faster learning narrow this range and shift it toward more accurate AIs, but cannot remove it. Expert deskilling starts as $P _ { A I } > a _ { N } - \lambda / \gamma$ , whatever the horizon, learning or decay.

Running example. With the competent AI, Anna at .50 accepts. Reaching .60 has some utility, but not enough to cover the gap of.17. At .40, even a step to .50 leaves Anna where accepting remains rational, so $\dot { V } ( . 4 0 ) = V ( . 5 0 )$ and this step is worthless. Anna thus stays at .40 or slides there from .50, while output accuracy stays at .62. Since Anna is less accurate than the AI, appropriate reliance scores every acceptance as appropriate. With the near-expert AI, Sam beats the AI by .03, less than the effort cost of .05. Sam accepts on every task andfalls one level after 100 tasks on average, while output accuracy stays at .77. Appropriate reliance first labels these acceptances as overreliance, but as appropriate once Samfalls to .70.

Strategic behavior alone does not prevent the low-skill trap or expert deskilling of workers. Even a strategic worker may fail to acquire or retain skill. In the low-skill trap, a novice whose accuracy is far below that of the AI keeps accepting because the cost of practicing until independent work pays off exceeds the future payoff. In expert deskilling, an expert whose accuracy advantage over the AI no longer outweighs the effort cost accepts the AI’s answer on every task, regardless of how many future tasks. In both cases, output accuracy remains at the AI’s level, while the worker’s unaided accuracy declines.

## 5 Dynamic Incentives for Independent Solving

Strategic behavior can leave a worker in the low-skill trap and let the skill of an expert decay despite a high skill level. To answer our second research question, we therefore investigate next how incentives, i.e., monetary payments, can make independent work worthwhile for workers. Following Holstein et al. [33], we make three assumptions: first, we assume that there exists an organization that pays an incentive for correct independent answers; second, we assume that the organization can observe a worker’s skill and verify that an answer was produced without the AI; third, we assume a myopic worker for now, because a payment that incentivizes a myopic worker also incentivizes a strategic one (Appendix A.9).

Let $\theta ( p ) \geq 0$ be an incentive paid only for a correct independent answer at skill level p. We resolve ties in favor of solving. A myopic worker therefore solves if and only if the expected incentive covers the immediate payoff gap:

$$
p \theta ( p ) \geq \Delta u ( p ) .\tag{9}
$$

## 5.1 Why the Effort Incentive is Insufficient

Holstein et al. [33] pay $\lambda / p$ for a correct independent answer in a model with fixed skill. In our dynamic setting, we evaluate this effort incentive at the worker’s current skill, $\theta ^ { \mathrm { e f f } } ( p _ { t } ) = \lambda / p _ { t }$ . At current skill $p ,$ this changes the expected utility of solving to

$$
u _ { S } ^ { \mathrm { e f f } } ( p ) = \gamma p - \lambda + p \frac { \lambda } { p } = \gamma p .\tag{10}
$$

Put simply, the incentive payment changes the expected utility to become only the expected earnings. Since the utility from accepting the AI’s answer is $\gamma P _ { A I } \left( \mathrm { E q . } 3 \right)$ , a myopic worker therefore solves if and only $\mathrm { i f } p \geq P _ { A I }$ . The effort incentive thus restores appropriate reliance, as Holstein et al. [33] intend, as workers no longer accept a less accurate AI merely to avoid effort.

![](images/7ab7ee9c677cba7038839779152e25d0dc0a453c4856ef28c0f1b3087a060b62.jpg)

![](images/a1df49f466e08216518ba4c21cefab402e4be1974ab40eb42f38d0682bae2a59.jpg)

![](images/dbb12cda8822619bcb6a27062e07fa79476904188c2820c2bd451d5c038c2898.jpg)  
Figure 2: Expected worker skill over completed tasks for a myopic worker with no incentive (grey), the effort incentive of Holstein et al. [33] (red), or the learning incentive in Eq. (11) (blue). The dotted horizontal line marks AI accuracy. (A) With competent AI, only the learning incentive builds novice skill. (B) With near-expert AI, both incentives retain expert skill at .80, so their curves overlap. (C) With above-expert AI, only the learning incentive retains expert skill. In panels A and C, the effort and no-incentive curves overlap.

Nevertheless, when workers accept the AI’s answer, they forgo the opportunities to acquire skill. So while the effort incentive restores appropriate reliance, it may still lead to skill decay. Strategic behavior does not necessarily tackle that issue: when the effort incentive is paid, a strategic worker faces the low-skill trap and expert deskilling (Appendix B.2).

Running example. With the competent AI, Anna accepts at .40, .50 and .60 despite the effort incentive, because each level is below .62. Every acceptance is appropriate, but Anna stays in the low-skill trap (Figure 2A). With the near-expert AI, the effort incentive keeps Sam solving at .80 and thus prevents deskilling (Figure 2B). However, once the AI becomes more accurate than Sam, accepting pays more at every skill level, even with the incentive, and Sam deskills again (Figure 2C). The effort incentive thus raises the AI accuracy at which experts start to deskill from $a _ { N } - \lambda / \gamma$ to $a _ { N }$ , but nofurther.

The incentive paid for the effort restores appropriate reliance but not learning. The incentive proposed by Holstein et al. [33] pays the worker for each correct independent answer so that the effort cost is covered in expectation. A myopic worker then solves if and only if they are at least as accurate as the AI, so effort no longer causes overreliance. However, a worker who is less accurate than the AI never starts to practice, and experts deskil as soon as the AI becomes more accurate than they are.

## 5.2 A Learning Incentive

To make solving worthwhile below the AI’s accuracy, an incentive must therefore cover both the effort and the difference between the AI’s accuracy and the worker’s skill level. Since both enter the immediate payoff gap $\Delta u ( p )$ , solving Eq. (9) for $\theta ( p )$ yields the minimum nonnegative learning incentive with $[ x ] ^ { + } = \operatorname* { m a x } \{ x , 0 \}$

$$
\theta ^ { \mathrm { { m i n } } } ( p ) = \frac { [ \Delta u ( p ) ] ^ { + } } { p } = \frac { [ \lambda + \gamma ( P _ { A I } - p ) ] ^ { + } } { p } .\tag{11}
$$

Wherever $\Delta u ( p ) \geq 0 .$ , this incentive splits into the effort incentive and an adjustment for the accuracy gap:

$$
\theta ^ { \mathrm { m i n } } ( p ) = \underbrace { \frac { \lambda } { p } } _ { \mathrm { e f f o r t i n c e n t i v e } } + \underbrace { \frac { \gamma ( P _ { A I } - p ) } { p } } _ { \mathrm { a c c u r a c y g a p } } .\tag{12}
$$

Below the AI’s accuracy, the second term compensates for the correct answers the worker forgoes by solving, so that the worker practices and can acquire skill. Above it, the worker’s own accuracy advantage covers part of the effort, so the second term is negative and the incentive falls below the effort incentive. The learning incentive therefore shrinks as more skills are acquired and vanishes once the accuracy advantage outweighs the effort, which also spares the organization from paying workers who would solve anyway. Appendix A.9 derives Eqs. (9)–(12).

Running example. With the competent AI, Anna needs an expected incentive of.27 per task at .40, .17 at .50, and .07 at .60, of which .05 covers effort and the rest the accuracy gap. At .70, solving pays without an incentive. With the near-expert AI, Sam at .80 needs only .02 instead ofthe effort incentive’s .05.

The learning incentive compensates for both the effort and the accuracy gap of the worker to the AI. To make independent work worthwhile for a worker who is less accurate than the AI, the organization must compensate the worker for both the effort and the correct answers the worker forgoes by not accepting the AI’s answer. This learning incentive is highest for novices and decreases as the worker acquires skill. For a worker who is more accurate than the AI, the learning incentive is lower than the effort incentive, and it ends at the skill level at which independent work is rational without an incentive.

## 5.3 Target Skill Levels

Paying the learning incentive at every skill level would also pay for practice the organization may not need. We therefore let the organization choose a target skill level $p ^ { \star } \in \bar { \mathcal { P } }$ , which can stand for the foundational skill that more advanced work requires. The organization requires independent solving at or below the target and leaves choices above it unsupported. The minimum incentive per correct independent answer that implements this policy for myopic workers is

$$
\theta ( p ; p ^ { \star } ) = \left\{ \begin{array} { l l } { \displaystyle \frac { [ \lambda + \gamma ( P _ { A I } - p ) ] ^ { + } } { p } , } & { p \leq p ^ { \star } , } \\ { 0 , } & { p > p ^ { \star } . } \end{array} \right.\tag{13}
$$

For a novice below the target, the target is a goal to acquire. For an expert at or above it, the target is a floor to retain. Whether payments end depends on whether the induced policy eventually reaches levels at which solving continues without support.

Acquiring a target. Below the target, the incentive makes the worker solve at every level, so the worker can acquire skill, but skill never decays. With $\phi _ { u } > 0$ , the worker therefore reaches the target almost surely, after $r / \phi _ { u }$ independent tasks on average when starting r levels below it. Reaching the target does not by itself end the payments, because the worker accepts again just above it. Payments end for good only once the worker reaches a level at which solving pays without an incentive. If that level lies one step above the target, the incentive keeps the worker practicing at the target until the step occurs, and the worker then keeps solving on their own. If that level lies further above, the worker accepts just above the target, slides back and triggers the incentive anew, so payments recur indefinitely. Payments thus end if and only if solving pays without an incentive one level above the target (Appendix A.9).

Running example. With the competent AI, a target of .60 bridges Anna to independence. The incentive supports practice up to .60, and Anna eventually reaches .70, where solving pays without an incentive $( . 7 0 \geq . 6 7 )$ . Starting from .40, the expected total spending is 25.50, after which payments stop (Appendix A.9). A target of .50 falls short. After reaching .60, Anna accepts again, falls back to .50 and triggers the incentive anew, so payments recur indefinitely. A higher target can thus turn recurring payments into temporary support, although it need not cost less over every finite horizon.

Retaining a target. Keeping skill at the target requires continued practice, but not necessarily continued payments. Whenever skill sits at the target, the incentive makes the worker solve. Since solving never lowers skill and accepting lowers it by at most one level, skill cannot fall below $p ^ { \star }$ while the incentive is in place. Payments continue only where solving does not pay on its own. At the highest target, $p ^ { \star } = a _ { N }$ , no further learning is possible, so if $\Delta u ( a _ { N } ) > 0$ , the worker needs a positive expected incentive on every task for as long as the skill is to be kept.

A lower target lets skill decay before the incentive restarts practice, but it can cost more in the long run. The incentive then pays repeatedly to acquire skill that was allowed to decay, and the larger payoff gap at the lower level makes each supported task more expensive. Appendix A.9 gives the condition under which the higher target costs less.

Running example. With the near-expert AI, keeping Sam at .80 costs .02 per task in expectation, still paid onlyfor correct independent answers. With a target of.70, Sam accepts untilfalling to .70, practices with the incentive until returning to .80 and then accepts again. Each supported task at .70 costs .12 instead of.02, and the long-run payment is .04 per task. Here, the higher target thus retains more skill at half the long-run cost.

Incentives for acquiring skill can be temporary, whereas retaining skill can require continuous incentives. Support to acquire a skill target ends once the worker reaches a level at which independent work is rational without incentive; otherwise skill repeatedly decays and incentives need to recur. Thus, retaining a skill level at which independent work is not rational requires incentives for as long as the skill is needed.

## 6 Discussion

Our primary theoretical contribution is to extend the existing accept-or-solve framework [7, 33] from a single task to a sequence of tasks in which each choice changes the skill of a worker. While research on AI-assisted decision-making has evaluated reliance within a task [32, 51], we show that the same choices determine the skill workers bring to future tasks. A myopic worker solves only when their accuracy advantage over the AI outweighs the effort, whereas a strategic worker may solve even when the AI is more accurate, a choice that appropriate reliance classifies as underreliance (RQ1). In line with prior dynamic models [3, 36], we show that strategic behavior does not prevent the low-skill trap or expert deskilling. In both cases, output accuracy remains at the level of the AI while skill decays. This may explain the empirically observed deskilling [17, 52]: when workers are rewarded for correct answers, accepting AI answers can be rational even though skill decays, and the decline can go unnoticed [44]. Since such effects unfold over many tasks and are hard to observe in short experiments, our model makes them analyzable before longitudinal evidence exists.

Our incentive analysis subsequently shows that paying for correct independent solving does not by itself build skill (RQ2). The effort incentive of Holstein et al. [33] restores appropriate reliance but not learning. The learning incentive, by contrast, additionally compensates for the accuracy gap between the worker and the AI, so that novices start to practice and experts retain their skill. Support to acquire a target skill level can end once independent solving pays without an incentive, whereas retaining a skill level at which independent work does not pay requires continuous incentives. This shows that incentive design for AI-assisted work must target the future skill of the worker rather than the accuracy of the current answer alone.

## 6.1 Appropriate Reliance and Skill

Since appropriate reliance compares only current accuracies, it labels a novice’s acceptances in the low-skill trap as appropriate and a strategic worker’s practice below the AI’s accuracy as underreliance. Once an expert falls below the AI’s accuracy, it also labels the expert’s remaining skill decay as appropriate. As D’Amour et al. [22] showed for fairness, short-term evaluations of workers’ accept-or-solve choices miss long-run consequences. We therefore call for extending appropriate reliance to account for how a decision changes a worker’s future skill.

Empirical studies of human–AI decision-making mostly measure output accuracy [40]. As our model shows, such studies can report stable output accuracy while the unaided accuracy that workers need for oversight and advanced work decays unnoticed. Evaluations of human–AI collaboration should therefore state their horizon and track unaided accuracy alongside output accuracy.

## 6.2 Incentives for Learning

With AI assistance, skill acquisition is becoming an explicit cost for organizations, which is new since acquiring skill has largely occurred as a by-product of assigned tasks [11]. When AI is used for these tasks, as in robotic surgery [9], this by-product disappears. Medical residency illustrates an alternative: residents are paid to do the work through which they learn, and their slower work is accepted as the price of future skill. Organizations can transfer this model as an AI-era apprenticeship that pays for the practice AI-assisted work no longer provides. Since the expected payment to acquire a target falls as the learning probability rises (Eq. (19)), feedback and well-chosen practice tasks that speed learning also reduce the incentive an organization must pay. Since retaining a higher-skill target can cost less than repeatedly restoring a lower one, organizations may need to budget differently and compare targets by long-run spending. Strategic workers never cost more than myopic workers under the same target, so organizations that help workers value skill on future tasks may also pay less.

Since the learning incentive compensates for both the effort and the reward forgone by not accepting AI answers (Eq. (11)), an activity that lets workers acquire skill while forgoing less reward would need a smaller incentive. Evidence on whether such AI-supported activities lead to skill acquisition is mixed: answering before seeing the AI’s answer improved immediate decisions but not learning [29], whereas explanations that contrast the AI’s answer with a predicted human choice improved learning more than explanations that only justify the AI’s answer [15]. Researchers should therefore identify which activities lead to skill acquisition at the lowest forgone reward. This includes practice whose answer does not replace the AI’s answer, such as simulated cases or independent attempts submitted only for review. Such practice forgoes no reward on the current task, so its incentive would need to cover only the effort cost.

While a worker is less accurate than the AI, promotion rankings based on output accuracy favor colleagues who accept the AI’s answer. The organization then pays for learning through one part of its reward system and penalizes it through another. Protected learning assignments and appraisals that recognize skill acquisition could ease this conflict. Whatever construct an appraisal measures, whether output accuracy, unaided accuracy, or professional judgment, workers will optimize for it [37].

## 6.3 Governing Skill Acquisition and Retention

Effective oversight under the EU AI Act requires adequate epistemic access [54]. Such policies require and presume that workers can perform the tasks [31]. Yet, our model shows that the skill on which oversight depends can decay while output accuracy stays stable. This suggests that oversight mandates should cover retaining the skill of overseers, not only assigning oversight. The EU AI Act already requires deployers of high-risk AI systems to provide human overseers with the necessary competence, training, and support [26, Art. 26(2)], but leaves open how deployers retain this skill once AI-assisted work no longer exercises it. Our learning incentive offers one mechanism for doing so, complementing automation research, which has long argued that workers, such as pilots, must regularly practice manual control, on the job or in simulators, to retain the skill needed to take over from automation [6].

Workers carry transferable skills to future employers, so employers may not capture the full returns of training, although labor-market frictions can let them benefit from investments in general skills [1]. Conversely, organizations can offer opportunities for skill acquisition to retain workers, since workers may stay where they can keep acquiring skill. Worker mobility can also lead organizations to offer skill-building opportunities mainly to workers who are already skilled [53]. Apprenticeships, sectoral training funds, and public co-funding can share these costs without restricting mobility or leaving less-skilled workers behind.

Existing institutions offer a precedent for governing such investments. German works councils, for example, codetermine in-company training when employer measures leave the skills of employees insufficient for their changed work, and they co-determine the introduction of technical devices that monitor conduct or performance [18, §§ 87(1) No. 6, 97(2)]. Implementations of our mechanism engage both rights, since they require assessing skill and verifying that answers were produced without the AI.

## 6.4 Fairness, Accountability, and Power

The learning incentive reverses a familiar pattern: worker pay usually rises with skill, whereas the learning incentive is largest for the least skilled worker. Since the incentive does not reward lower skill but compensates for the effort and the reward a worker forgoes while practicing below the AI’s accuracy. Without it, the starting skill of a worker determines whether learning pays at all, and novices remain in the low-skill trap not because they cannot learn but because learning is not rational. A larger incentive for novices thus equalizes the opportunity to acquire skill rather than the outcome, much as subsidizing the cost of investment can prevent disparate equilibria when individuals invest rationally in their qualifications [43]. Colleagues who receive less pay per task for the same work may nevertheless perceive the incentive as unfair, and an organization with a fixed budget has a calculable reason to support workers who are already skilled.

The costs of learning fall partly on people whom the incentive does not compensate. In our example of acquiring a target, a novice supported from .40 to .60 submits about 18 more incorrect answers in expectation than the AI would. The clients affected by these errors are not necessarily those who later benefit from the skill of the worker. Without support for retaining skill, workers in turn risk being held responsible for failures of systems over which they have limited control [24]

In our model, an external party, like an organization, chooses the skill target, observes the skill, and sets the incentive. With this, they decide on which skills workers require or retain and up to which skill level. Letting a skill decay might be legitimate, but it becomes harmful when the interests of workers contradict those of the organization. Since workers contribute the effort, are potentially affected by their organization’s skill assessments, and may value skills differently, they should participate in setting targets and deciding how assessments are used, for example through works councils. Skill assessments could inform discipline, and because the incentive is smaller at higher skill levels, workers may benefit from appearing less skilled, while assessments robust to such manipulation can impose their costs unequally [35, 45].

## 6.5 Limitations and Future Research

Our numerical examples do not establish how often the low-skill trap or expert deskilling occurs, how long workplace learning takes, or which payments workers would require in real-world settings. Longitudinal studies should estimate effort costs, task rewards, learning and decay probabilities, and the horizons over which workers expect to benefit. They should also test whether incentives induce practice, whether that practice builds and retains the intended skill, and how payments and workload evolve. Since acquiring a target assumes indefinitely continuing tasks, our results do not guarantee reaching a target within a fixed-term employment period.

In our model, skill is acquired only through independent solving and represents unaided accuracy in one task family. If workers also acquire skill through engaged AI use, the low-skill trap may weaken, and the learning incentive could pay for skill acquisition that AI-assisted work would provide anyway. Retaining a target relies on the assumption that solving never decays skill and on accepting decays it by at most one skill level. Future research should consider learning and decay rates that vary across tasks and over time, as well as declines larger than one level. Models with several linked skills could examine whether practicing foundational skills lowers the cost of acquiring advanced ones. Such models could also capture collaboration with AI as a skill of its own that makes the original skill less relevant. In writing, for example, directing and editing drafts with AI assistance may matter more than drafting text unaided.

The minimum incentive applies to myopic workers paid for correct independent answers, with observable skill, verifiable independence, ties resolved in favor of solving, and known effort costs and task rewards, and workers are assumed to know their own accuracy. Since unsuccessful solving attempts might also teach, payments for attempts, protected practice time, and non-monetary incentives should be studied in the future.

Finally, we analyze a single worker and measure the organization only through incentive spending. This spending omits the accuracy forgone during practice and the value of the acquired skill, so our comparisons of targets do not identify a welfare-maximizing target. Extending the model to an organizational objective and to heterogeneous workers, whose learning probabilities and starting levels determine the cost of reaching a target, remains future work.

## 7 Conclusion

Today’s output accuracy provides an incomplete view of how AI-assisted work will shape tomorrow’s workforce. We extended an established accept-or-solve model to a sequence of tasks in which solving independently can let the worker acquire skill, whereas accepting the AI’s answer can let skill decay. Strategic behavior can make independent work worthwhile even when the AI is more accurate, although appropriate reliance classifies this choice as underreliance. However, strategic behavior alone does not guarantee that workers will acquire or retain skill. Novices can remain in a low-skill trap, and experts can deskill, while output accuracy remains at the AI’s level. The effort incentive restores appropriate reliance, but not learning. The learning incentive we derive additionally compensates for the accuracy gap between the worker and the AI. Support to acquire a target skill level can end once independent work pays without an incentive, whereas retaining a skill level at which independent work does not pay requires continuous incentives. Just as no astronaut can stay strong by waiting for gravity to return, no organization can keep its experts by waiting for AI to fail. Exercise must be scheduled while work is still being done, so that experts retain their skill and novices acquire it in the first place. Space agencies only began scheduling exercises after astronauts returned to Earth too weak to stand. Organizations adopting AI cannot afford to learn this lesson the same way. The loss of skill remains hidden until the AI fails and someone must take over. Organizations that wait until then will find that the necessary skills are gone, and there is no one left to teach them.

## Generative AI Usage Statement

The authors used the following generative AI tools during manuscript preparation: Claude Opus 5.5 (Anthropic), GPT6.1-Sol and GPT6-Astra (OpenAI), DeepL Write, and Grammarly. We used these tools for literature search, drafting text and language editing. We also used them to write code for the model, robustness checks, figures, and to check the mathematical derivations. The authors reviewed and edited all AI-generated content and take full responsibility for the content of this paper.

## Ethical Considerations and Adverse Impacts

This work is theoretical and thus involves no human participants, no personal data, or deployed system. The numerical examples use illustrative parameter values, not empirical estimates.

## References

[1] Daron Acemoglu and Jörn-Steffen Pischke. 1999. Beyond Becker: Training in Imperfect Labour Markets. The Economic Journal 109, 453 (1999), F112–F142. doi:10.1111/1468-0297.00405

[2] Callen Anthony. 2021. When Knowledge Work and Analytical Technologies Collide: The Practices and Consequences of Black Boxing Algorithmic Technologies. Administrative Science Quarterly 66, 4 (2021), 1173–1212. doi:10.1177/00018392211016755

[3] Ali Aouad, Thodoris Lykouris, and Huiying Zhong. 2026. Human-AI Productivity Paradoxes: Modeling the Interplay of Skill, Effort, and AI Assistance. arXiv:2605.11350 https://arxiv.org/abs/2605.11350

[4] Winfred Arthur, Jr., Winston Bennett, Jr., Pamela L. Stanush, and Theresa L. McNelly. 1998. Factors That Influence Skill Decay and Retention: A Quantitative Review and Analysis. Human Performance 11, 1 (1998), 57–101. doi:10.1207/s15327043hup1101\_3

[5] David Autor and Neil Thompson. 2025. Expertise. Journal of the European Economic Association 23, 4 (2025), 1203–1271. doi:10.1093/jeea/jvaf023

[6] Lisanne Bainbridge. 1983. Ironies of Automation. Automatica 19, 6 (1983), 775–779. doi:10.1016/ 0005-1098(83)90046-8

[7] Gagan Bansal, Besmira Nushi, Ece Kamar, Eric Horvitz, and Daniel S. Weld. 2021. Is the Most Accurate AI the Best Teammate? Optimizing AI for Teamwork. Proceedings ofthe AAAI Conference on Artificial Intelligence 35, 13 (2021), 11405–11414. doi:10.1609/aaai.v35i13.17359

[8] Hamsa Bastani, Osbert Bastani, Alp Sungu, Haosen Ge, Özge Kabakcı, and Rei Mariman. 2025. Generative AI without Guardrails Can Harm Learning: Evidence from High School Mathematics. Proceedings ofthe National Academy ofSciences 122, 26 (2025), e2422633122. doi:10.1073/pnas.2422633122

[9] Matthew Beane. 2019. Shadow Learning: Building Robotic Surgical Skill When Approved Means Fail. Administrative Science Quarterly 64, 1 (2019), 87–123. doi:10.1177/0001839217751692

[10] Gary S. Becker. 1964. Human Capital: A Theoretical and Empirical Analysis, with Special Reference to Education. Number 80 in NBER General Series. National Bureau of Economic Research (distributed by Columbia University Press), New York.

[11] Stephen Billett. 2001. Learning through Work: Workplace Affordances and Individual Engagement. Journal of Workplace Learning 13, 5 (2001), 209–214. doi:10.1108/EUM0000000005548

[12] Robert A. Bjork. 1994. Memory and Metamemory Considerations in the Training of Human Beings. In Metacognition: Knowing about Knowing, Janet Metcalfe and Arthur P. Shimamura (Eds.). MIT Press, Cambridge, MA, 185–206. doi:10.7551/mitpress/4561.003.0011

[13] Erik Brynjolfsson, Bharat Chandar, and Ruyu Chen. 2026. Canaries in the Coal Mine? Six Facts about the Recent Employment Effects of Artificial Intelligence. Working Paper. Stanford Digital Economy Lab. https: //digitaleconomy.stanford.edu/app/uploads/2026/08/Canaries\_August2026.pdf

[14] Zana Buçinca, Maja Barbara Malaya, and Krzysztof Z. Gajos. 2021. To Trust or to Think: Cognitive Forcing Functions Can Reduce Overreliance on AI in AI-assisted Decision-making. Proceedings ofthe ACM on Human-Computer Interaction 5, CSCW1, Article 188 (2021), 21 pages. doi:10.1145/3449287

[15] Zana Buçinca, Siddharth Swaroop, Amanda E. Paluch, Finale Doshi-Velez, and Krzysztof Z. Gajos. 2025. Contrastive Explanations That Anticipate Human Misconceptions Can Improve Human Decision-Making Skills. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25). Association for Computing Machinery, New York, NY, USA, Article 1024, 25 pages. doi:10.1145/3706598.3713229

[16] Zana Buçinca, Siddharth Swaroop, Amanda E. Paluch, Susan A. Murphy, and Krzysztof Z. Gajos. 2026. Offline Reinforcement Learning for Adaptive Support in AI-Assisted Decision-Making. ACM Transactions on Computer-Human Interaction 33, 4 (2026), 1–43. doi:10.1145/3815418 Preprint: arXiv:2403.05911 (2024).

[17] Krzysztof Budzyn, Marcin Roma´ nczyk, Diana Kitala, Paweł Kołodziej, Marek Bugajski, Hans O. Adami, Johannes´ Blom, Marek Buszkiewicz, Natalie Halvorsen, Cesare Hassan, Tomasz Romanczyk, Øyvind Holme, Krzysztof´ Jarus, Shona Fielding, Melina Kunar, Maria Pellise, Nastazja Pilonis, Michał Filip Kaminski, Mette Kalager,´ Michael Bretthauer, and Yuichi Mori. 2025. Endoscopist Deskilling Risk after Exposure to Artificial Intelligence in Colonoscopy: A Multicentre, Observational Study. The Lancet Gastroenterology & Hepatology 10, 10 (2025), 896–903. doi:10.1016/S2468-1253(25)00133-5

[18] Bundesrepublik Deutschland. 2001. Betriebsverfassungsgesetz (BetrVG). Works Constitution Act, as published 25 September 2001 (BGBl. I S. 2518), as amended. Retrieved November 3, 2026 from https: //www.gesetze-im-internet.de/betrvg/

[19] Ilias Chalkidis and Anders Søgaard. 2026. Brainrot: Deskilling and Addiction are Overlooked AI Risks. Associa tion for Computing Machinery, 3005–3028. doi:doi.org/10.1145/3805689.381230

[20] Allan Collins, John Seely Brown, and Susan E. Newman. 1989. Cognitive Apprenticeship: Teaching the Crafts of Reading, Writing, and Mathematics. In Knowing, Learning, and Instruction: Essays in Honor of Robert Glaser, Lauren B. Resnick (Ed.). Lawrence Erlbaum Associates, Hillsdale, NJ, 453–494. doi:10.4324/ 9781315044408-14

[21] Albert T. Corbett and John R. Anderson. 1994. Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge. User Modeling and User-Adapted Interaction 4, 4 (1994), 253–278. doi:10.1007/BF01099821

[22] Alexander D’Amour, Hansa Srinivasan, James Atwood, Pallavi Baljekar, D. Sculley, and Yoni Halpern. 2020. Fairness Is Not Static: Deeper Understanding of Long Term Fairness via Simulation Studies. In Proceedings ofthe 2020 Conference on Fairness, Accountability, and Transparency (FAT\* ’20). Association for Computing Machinery, New York, NY, USA, 525–534. doi:10.1145/3351095.3372878

[23] Fabrizio Dell’Acqua, Edward McFowland, III, Ethan Mollick, Hila Lifshitz, Katherine C. Kellogg, Saran Rajendran, Lisa Krayer, François Candelon, and Karim R. Lakhani. 2026. Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of Artificial Intelligence on Knowledge Worker Productivity and Quality. Organization Science 37, 2 (2026), 403–423. doi:10.1287/orsc.2025.21838

[24] Madeleine Clare Elish. 2019. Moral Crumple Zones: Cautionary Tales in Human-Robot Interaction. Engaging Science, Technology, and Society 5 (2019), 40–60. doi:10.17351/ests2019.260

[25] K. Anders Ericsson, Ralf Th. Krampe, and Clemens Tesch-Römer. 1993. The Role of Deliberate Practice in the Acquisition of Expert Performance. Psychological Review 100, 3 (1993), 363–406. doi:doi.org/10.1037/ 0033-295X.100.3.363

[26] European Parliament and Council of the European Union. 2024. Regulation (EU) 2024/1689 of 13 June 2024 Laying Down Harmonised Rules on Artificial Intelligence (Artificial Intelligence Act). Official Journal of the European Union, OJ L, 2024/1689, 12 July 2024. Retrieved November 1, 2026 from http://data.europa. eu/eli/reg/2024/1689/oj

[27] Drew Fudenberg and Luis Rayo. 2019. Training and Effort Dynamics in Apprenticeship. American Economic Review 109, 11 (2019), 3780–3812. doi:10.1257/aer.20171939

[28] Robert M. Gagné. 1962. The Acquisition of Knowledge. Psychological Review 69, 4 (1962), 355–365. doi:doi. org/10.1037/h0042650

[29] Krzysztof Z. Gajos and Lena Mamykina. 2022. Do People Engage Cognitively with AI? Impact of AI Assistance on Incidental Learning. In Proceedings of the 27th International Conference on Intelligent User Interfaces (IUI ’22). Association for Computing Machinery, 794–806. doi:10.1145/3490099.3511138

[30] Luis Garicano and Luis Rayo. 2017. Relational Knowledge Transfers. American Economic Review 107, 9 (2017), 2695–2730. doi:10.1257/aer.20160194

[31] Ben Green. 2022. The Flaws of Policies Requiring Human Oversight of Government Algorithms. Computer Law & Security Review 45 (2022), 105681. doi:10.1016/j.clsr.2022.105681

[32] Ziyang Guo, Yifan Wu, Jason D. Hartline, and Jessica Hullman. 2024. A Decision Theoretic Framework for Measuring AI Reliance. In Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’24). Association for Computing Machinery, 221–236. doi:10.1145/3630106.3658901

[33] Joshua Holstein, Patrick Hemmer, Gerhard Satzger, and Wei Sun. 2026. When Thinking Pays Off: Incentive Alignment for Human-AI Collaboration. In Proceedings ofthe 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’26). Association for Computing Machinery, New York, NY, USA, 1139–1165. doi:10. 1145/3805689.3812250

[34] Seyed M. Hosseini Maasoum and Guy Lichtinger. 2025. Generative AI as Seniority-Biased Technological Change: Evidence from U.S. Résumé and Job Posting Data. SSRN Working Paper 5425555. doi:10.2139/ssrn.5425555

[35] Lily Hu, Nicole Immorlica, and Jennifer Wortman Vaughan. 2019. The Disparate Effects of Strategic Manipulation. In Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* ’19). Association for Computing Machinery, 259–268. doi:10.1145/3287560.3287597

[36] Lingxiao Huang and Nisheeth K. Vishnoi. 2026. Path Dependence under Adaptive AI Delegation. In Advances in Neural Information Processing Systems (NeurIPS 2026). doi:10.48550/arXiv.2603.02950 arXiv:2603.02950.

[37] Abigail Z. Jacobs and Hanna Wallach. 2021. Measurement and Fairness. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’21). Association for Computing Machinery, 375–385. doi:10.1145/3442188.3445901

[38] Rishub Jain, Sophie Bridgers, Lili Janzer, Rory Greig, Tian Huey Teh, and Vladimir Mikulik. 2026. Human-AI Complementarity: A Goal for Amplified Oversight. Association for Computing Machinery, 249–279. doi:doi. org/10.1145/3805689.3812308

[39] Manu Kapur. 2008. Productive Failure. Cognition and Instruction 26, 3 (2008), 379–424. doi:10.1080/ 07370000802212669

[40] Vivian Lai, Chacha Chen, Alison Smith-Renner, Q. Vera Liao, and Chenhao Tan. 2023. Towards a Science of Human-AI Decision Making: An Overview of Design Space in Empirical Human-Subject Studies. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’23). Association for Computing Machinery, 1369–1385. doi:10.1145/3593013.3594087

[41] Jean Lave and Etienne Wenger. 1991. Situated Learning: Legitimate Peripheral Participation. Cambridge University Press, Cambridge. doi:doi.org/10.1017/CBO9780511815355

[42] John D. Lee and Katrina A. See. 2004. Trust in Automation: Designing for Appropriate Reliance. Human Factors 46, 1 (2004), 50–80. doi:10.1518/hfes.46.1.50.30392

[43] Lydia T. Liu, Ashia Wilson, Nika Haghtalab, Adam Tauman Kalai, Christian Borgs, and Jennifer Chayes. 2020. The Disparate Equilibria of Algorithmic Decision Making when Individuals Invest Rationally. In Proceedings ofthe 2020 Conference on Fairness, Accountability, and Transparency (FAT\* ’20). Association for Computing Machinery, 381–391. doi:10.1145/3351095.3372861

[44] Brooke N. Macnamara, Ibrahim Berber, M. Cenk Çavu¸soglu, Elizabeth A. Krupinski, Naren Nallapareddy,˘ Noelle E. Nelson, Philip J. Smith, Amy L. Wilson-Delfosse, and Soumya Ray. 2024. Does Using Artificial Intelligence Assistance Accelerate Skill Decay and Hinder Skill Development without Performers’ Awareness? Cognitive Research: Principles and Implications 9, 1 (2024), 46. doi:10.1186/s41235-024-00572-8

[45] Smitha Milli, John Miller, Anca D. Dragan, and Moritz Hardt. 2019. The Social Cost of Strategic Classification. In Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* ’19). Association for Computing Machinery, 230–239. doi:10.1145/3287560.3287576

[46] Chiara Natali, Luca Marconi, Leslye Denisse Dias Duran, and Federico Cabitza. 2025. AI-Induced Deskilling in Medicine: A Mixed-Method Review and Research Agenda for Healthcare and Beyond. Artificial Intelligence Review 58, 11 (2025), 356. doi:10.1007/s10462-025-11352-1

[47] Martin L. Puterman. 1994. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons. doi:10.1002/9780470316887

[48] Tapani Rinta-Kahila, Esko Penttinen, Antti Salovaara, Wael Soliman, and Joona Ruissalo. 2023. The Vicious Circles of Skill Erosion: A Case Study of Cognitive Automation. Journal of the Association for Information Systems 24, 5 (2023), 1378–1412. doi:10.17705/1jais.00829

[49] Evan F. Risko and Sam J. Gilbert. 2016. Cognitive Offloading. Trends in Cognitive Sciences 20, 9 (2016), 676–688. doi:10.1016/j.tics.2016.07.002

[50] Max Schemmer, Andrea Bartos, Philipp Spitzer, Patrick Hemmer, Niklas Kühl, Jonas Liebschner, and Gerhard Satzger. 2023. Towards Effective Human-AI Decision-Making: The Role of Human Learning in Appropriate Reliance on AI Advice. In Proceedings ofthe 44th International Conference on Information Systems (ICIS 2023). Association for Information Systems, Hyderabad, India. https://aisel.aisnet.org/icis2023/hti/hti 14

[51] Max Schemmer, Niklas Kühl, Carina Benz, Andrea Bartos, and Gerhard Satzger. 2023. Appropriate Reliance on AI Advice: Conceptualization and the Effect of Explanations. In Proceedings ofthe 28th International Conference on Intelligent User Interfaces (IUI ’23). Association for Computing Machinery, 410–422. doi:10.1145/3581641. 3584066

[52] Judy Hanwen Shen and Alex Tamkin. 2026. How AI Impacts Skill Formation. arXiv:2601.20245 https: //arxiv.org/abs/2601.20245v2

[53] Simrita Singh, Naireet Ghosh, and Tinglong Dai. 2026. Managing the Human Fallback: Skill Investment Under Improving AI and Worker Mobility. arXiv:2606.29111 https://arxiv.org/abs/2606.29111

[54] Sarah Sterz, Kevin Baum, Sebastian Biewer, Holger Hermanns, Anne Lauber-Rönsberg, Philip Meinel, and Markus Langer. 2024. On the Quest for Effectiveness in Human Oversight: Interdisciplinary Perspectives. In Proceedings ofthe 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’24). Association for Computing Machinery, 2495–2507. doi:10.1145/3630106.3659051

[55] Shang Wu, Catarina Belem, Shuyuan Fu, Mark Steyvers, and Padhraic Smyth. 2026. How AI Assistance Affects Human Skill Development: A Study of Learning with Logic Puzzles. In Proceedings of the 2026 ACM Conference on Human-AI Complementarity and Alignment (HCOMP ’26). Association for Computing Machinery, 88–96. doi:10.1145/3834580.3838741

[56] Sumin Yu and Taesup Moon. 2026. Who Will Become the Next Senior? How Generative AI Erodes the Development Pathway in Software Engineering. In Proceedings ofthe 2026 AAAI/ACM Conference on AI, Ethics, and Society (AIES ’26, Vol. 9). AAAI Press, 2825–2837. doi:10.1609/aies.v9i4.43353

## Appendix

## A Derivations of the model equations

This appendix derives every equation in Sections 3–5 step by step and shows how the numbers in the running example follow from them. Table 2 shows the notation, and Table 3 shows where each equation of the main text is derived in the appendix. All numbers are exact expectations computed from the transition rules, without simulation.

Table 2: Notation and running-example values.
<table><tr><td>Symbol</td><td>Meaning</td><td>Running example</td></tr><tr><td> $\mathcal { P } = \{ a _ { 1 } < \cdots < a _ { N } \}$ </td><td>skill levels (unaided accuracy)</td><td> $\{ . 4 0 , . 5 0 , . 6 0 , . 7 0 , . 8 0 \} , N = 5$ </td></tr><tr><td> $p , p ^ { + } , p ^ { - }$ </td><td>current level and its upper and lower neighbor</td><td> $\begin{array} { r } { \mathtt { e . g . } p = . 6 0 , p ^ { + } = . 7 0 , } \end{array}$   $p ^ { - } = . 5 0$ </td></tr><tr><td> $P _ { A I }$ </td><td>accuracy of the AI</td><td>competent .62, near-expert .77, above-expert .82</td></tr><tr><td>γ</td><td>reward for a correct final answer</td><td>1</td></tr><tr><td>λ</td><td>effort cost of solving independently</td><td>.05</td></tr><tr><td> $\phi _ { u }$ </td><td>probability of a learning step after solving</td><td>.02</td></tr><tr><td> $\phi _ { d }$   $\delta$ </td><td>probability of a decay step after accepting</td><td>.01</td></tr><tr><td> $\Delta u ( p )$ </td><td>discount factor (probability that another task follows) immediate payoff gap, Eq. (4)</td><td> $. 9 9 , \mathrm { { s o } } H = 1 / ( 1 - \delta ) = 1 0 0$   $\mathbf { e . g . } \Delta u ( . 6 0 ) = . 0 7$  with the</td></tr><tr><td></td><td></td><td>competent AI</td></tr><tr><td> $V ( p )$   $\theta ( p )$ </td><td>lifetime value, Eq. (6) incentive per correct independent answer</td><td></td></tr></table>

Table 3: Where each equation of the main text is derived.
<table><tr><td>Main text</td><td>Content</td><td>Appendix</td></tr><tr><td>Eqs. (1), (2)</td><td>skill levels as a Markov chain, expected next-task accuracy</td><td>A.1</td></tr><tr><td>Table 1, Eqs. (3), (4)</td><td>utility of one task, immediate payoff gap</td><td>A.2</td></tr><tr><td>Eq. (5)</td><td>myopic decision rule</td><td>A.3</td></tr><tr><td>Eq. (6)</td><td>lifetime value, meaning of δ</td><td>A.4</td></tr><tr><td>Eqs. (7), (8)</td><td>Bellman equation, strategic decision rule</td><td>A.5</td></tr><tr><td></td><td>how the model is solved</td><td>A.6</td></tr><tr><td>Figure 1</td><td>closed form of the strategic choice, policy regions</td><td>A.7</td></tr><tr><td>Section 4.3</td><td>low-skill trap, expert deskilling</td><td>A.8</td></tr><tr><td>Eqs. (9)–(13)</td><td>incentives and their costs</td><td>A.9</td></tr><tr><td>Section 4.2, running example</td><td>strategic return of independent work</td><td>A.10</td></tr></table>

## A.1 Skill dynamics as a Markov chain

Transition rule. Skill takes one of the N levels in Eq. (1). After each task, skill either stays on the same level or moves to an adjacent level. Which moves are possible depends on the chosen action:

$$
\mathrm { S o l v e : } \quad p _ { t + 1 } = \left\{ \begin{array} { l l } { p ^ { + } } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } \phi _ { u } , } \\ { p } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 - \phi _ { u } , } \end{array} \right. \qquad \mathrm { A c c e p t : } \quad p _ { t + 1 } = \left\{ \begin{array} { l l } { p ^ { - } } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } \phi _ { d } , } \\ { p } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 - \phi _ { d } . } \end{array} \right.
$$

At the highest skill level, a learning step is not possible, so $p ^ { + } = a _ { N }$ and the worker stays. At the lowest level, the same holds for a decay step, $p ^ { - } = a _ { 1 }$ . Solving therefore never lowers skill, and accepting never raises it. Figure 3A shows these transitions.

The transition rule makes skill a finite-state Markov chain whose transitions the worker controls through the action on each task.

Finite, ordered levels. Skill is the worker’s unaided accuracy, divided into N ordered levels. Knowledge tracing [21] uses the simplest such division, with only two states in which a skill is either learned or not. More levels allow accuracy to rise and fall gradually. Appendix B shows that the results do not depend on the number or spacing of the levels.

Memoryless transitions. The next level depends only on the current level and the current action, not on how the worker reached that level. The current level thus summarizes everything about past practice that matters for the future. The model therefore does not capture the effects of the order of past tasks, such as how practice is spaced over time.

One level per task. Each task moves skill by at most one level, so skill changes gradually rather than in jumps. A Markov chain with this property is a birth–death chain, the structure that Aouad et al. [3] also use. In their model, learning depends on effort and decay is exogenous. In our model, both directions depend on the worker’s choice on each task: only solving can raise skill, and only accepting can lower it.

Constant step probabilities. Knowledge tracing assumes that the probability of learning is the same at every practice opportunity [21]. We likewise keep $\phi _ { u }$ and $\phi _ { d }$ constant across levels. This keeps the model parsimonious, although the results for strategic workers do not require it: Appendix A.7 shows that the closed form extends to probabilities that differ between levels.

Holding boundaries. $\mathbf { A t } a _ { 1 }$ and $a _ { N }$ , a step that would leave the grid is replaced by staying. The self-loop at the boundary therefore has probability one under the corresponding action.

Together, these choices turn the strategic worker’s problem into a finite Markov decision process [47], with N states and two actions. All expectations can therefore be computed exactly from the transition matrices (Appendix A.6), without simulation, and the optimal policy has a closed form (Appendix A.7).

Chains under fixed policies. Once the worker’s action at each level is fixed, only one set of arcs remains at each level, and the resulting chain shows where skill ends up in the long run (Figure 3B–D). Under a policy that accepts below some level and solves from that level upward, no transition connects the highest accepting level with the lowest solving level. The chain then splits into two parts: from accepting levels, skill can only drift down to $a _ { 1 } .$ , and from solving levels it can only climb to $a _ { N }$ . Both $a _ { 1 }$ and $a _ { N }$ are absorbing. A worker who starts below the threshold therefore ends at $a _ { 1 } .$ , which is the low-skill trap (panel B). If every level accepts, $a _ { 1 }$ is the only absorbing level, which is expert deskilling (panel C). Under a target incentive (Eq. (13)) that leaves the worker accepting just above the target, skill eventually alternates between the target and the level above it (panel D). The long-run share of tasks spent at the target is $( 1 / \phi _ { u } ) / ( 1 / \phi _ { u } + 1 / \phi _ { d } ) = \phi _ { d } / ( \phi _ { u } + \phi _ { d } )$ , which sets the long-run cost of retaining it (Appendix A.9).

Expected waiting time for a step. After each solved task, a learning step occurs with the same probability $\phi _ { u } ,$ independently of earlier tasks. The number of solved tasks $K$ until the first step therefore follows a geometric distribution, $\bar { \operatorname* { P r } } ( K = k ) = ( 1 - \phi _ { u } ) ^ { k - 1 } \phi _ { u }$ for $k = 1 , 2 , \ldots$ , with mean

$$
\operatorname { \mathbb { E } } [ K ] = \sum _ { k \geq 1 } k \left( 1 - \phi _ { u } \right) ^ { k - 1 } \phi _ { u } = { \frac { 1 } { \phi _ { u } } } .\tag{14}
$$

The same argument gives an average of $1 / \phi _ { d }$ accepted tasks until a decay step. To climb r levels by solving, the worker waits r times in a row, so by linearity of expectation the climb takes $r / \phi _ { u }$ solved tasks on average.

Running example. With $\phi _ { u } = . 0 2$ and $\phi _ { d } = . 0 1$ , a developer who solves rises one level after 50 tasks on average, and one who acceptsfalls one level after 100 tasks on average. Climbingfrom .40 to .70 takes $3 \times 5 0 = 1 5 0$ solved tasks on average.

Expected next-task accuracy, $\mathbf { E q } . ( 2 )$ . After solving, the next-task accuracy is $p ^ { + }$ with probability $\phi _ { u }$ and $p$ otherwise. Its expectation is therefore $( 1 - \phi _ { u } ) p + \phi _ { u } p ^ { + }$ . After accepting, it is $( 1 - \phi _ { d } ) p + \phi _ { d } p ^ { - }$ . Subtracting the second from the first and collecting terms gives

$$
\left[ ( 1 - \phi _ { u } ) p + \phi _ { u } p ^ { + } \right] - \left[ ( 1 - \phi _ { d } ) p + \phi _ { d } p ^ { - } \right] = \phi _ { u } ( p ^ { + } - p ) + \phi _ { d } ( p - p ^ { - } ) .
$$

Both terms are nonnegative because $p ^ { + } \geq p \geq p ^ { - }$ . At the highest level, the first term vanishes because $p ^ { + } = p ,$ , leaving only the avoided decay. At the lowest level, the second term vanishes.

Running example. $A t p = . 6 0$ , solving rather than accepting raises the expected next-task accuracy by $. 0 2 \times . 1 0 +$ $. 0 1 \times . 1 0 = . 0 0 3$

## A.2 Expected utility of one task

Expected utilities, $\mathbf { E q } . \mathbf { \Gamma } ( 3 )$ . Each entry of Table 1 is weighted by the probability that the final answer is correct. Accepting submits the AI’s answer, which is correct with probability $P _ { A I } { \mathrm { : } }$

$$
u _ { A } = P _ { A I } \cdot \gamma + ( 1 - P _ { A I } ) \cdot 0 = \gamma P _ { A I } .
$$

Solving submits the worker’s own answer, which is correct with probability $p ,$ and costs λ in either case:

$$
u _ { S } ( p ) = p ( \gamma - \lambda ) + ( 1 - p ) ( - \lambda ) = \gamma p - \lambda .
$$

A Controlled chain: Solve (above) and Accept (below)

![](images/632e7fdceb5caf6123f9eaabd86e01e5d758053849a74ea7138d55eb45a52183.jpg)

B Low-skill trap: strategic worker $( H = 1 0 0 )$ , competent AI $( P _ { A I } = . 6 2 )$  
![](images/9b5c6fb9d9081a038a4f53267b040acb865055fcebef049a837e0c5f05f8d3d0.jpg)

C Expert deskilling: any worker, near-expert AI $( P _ { A I } = . 7 7 )$

![](images/a5a5e82686b455d89a75c670e8a7bbad994a96d02f4b252a28fcc10e61016475.jpg)

D Retaining target $p ^ { \star } = . 7 0 $ : myopic worker, near-expert AI $( P _ { A I } = . 7 7 )$

![](images/69c43aa571fab2322fce675d058813ecb183cd98cfb06f7255eaf483735f6d39.jpg)

$$
7 0 \mathrm { i s } \phi _ { d } / ( \phi _ { u } + \phi _ { d } )
$$

Figure 3: Skill dynamics as a Markov chain. (A) The controlled chain. If the worker solves, skill rises by one level with probability $\phi _ { u } ;$ if the worker accepts, it falls by one level with probability $\phi _ { d } ;$ otherwise it stays. At the boundaries, a step that would leave the grid is replaced by staying, so the corresponding self-loop has probability one. (B–D) Chains under fixed policies on the running-example grid. Fill marks the action at each level; self-loops are omitted. (B) A strategic worker with the competent AI accepts at .40 and .50 and solves above. No transition connects .50 and .60, so a worker starting at .50 can only drift to .40. (C) With the near-expert AI, every level accepts, and skill drifts to .40. (D) The target incentive for $p ^ { \star } = . 7 0$ makes the worker solve up to the target, while the worker accepts at .80. Skil eventually alternates between .70 and .80; the share of tasks at .70 sets the long-run cost of the target (Appendix A.9). Parameters as in Section 3.2.

Relation to the utilities of Bansal et al. [7]. Bansal et al. [7, Table 1] reward a correct answer with 1 and penalize an incorrect one with $- \beta ,$ , and solving also costs λ. Their four entries are 1 and −β for accepting and $1 - \lambda$ and $- \beta - \lambda$ for solving. Adding the constant $\bar { \beta }$ to every entry gives $1 + \beta$ and 0 for accepting and $1 \overset { \cdot } { + } \beta - \lambda$ and −λ for solving, which is Table 1 with $\gamma = 1 + \beta .$ . A constant added to every outcome raises the expected utility of both actions by the same amount, so it does not change the comparison between them. Over a sequence of tasks, it adds the same constant $\beta / ( 1 - \delta )$ to every lifetime value in Eq. (6), which again changes no decision.

Immediate payoff gap, Eq. (4). Subtracting the two expected utilities gives

$$
\Delta u ( p ) = u _ { A } - u _ { S } ( p ) = \gamma { \cal P } _ { A I } - ( \gamma p - \lambda ) = \lambda + \gamma ( { \cal P } _ { A I } - p ) .
$$

Dividing by $\gamma > 0$ does not change the sign, and $\Delta u ( p ) / \gamma = \lambda / \gamma + P _ { A I } - p .$ . Every decision therefore depends on λ and γ only through their ratio $\lambda / \gamma$ . The gap falls by γ for every unit of skill, so it is largest at the lowest level.

Running example. With the competent AI, $\Delta u ( p ) = . 0 5 + . 6 2 - p ,$ which $i s . 2 7 , . 1 7 , . 0 7 , - . 0 3$ , and −.13 at the five levels. With the near-expert AI, $\Delta u ( p ) = . 0 5 + . 7 7 - p ,$ which is .42, .32, .22, .12, and .02.

## A.3 The myopic decision rule

A myopic worker compares only today’s expected utilities and solves if $u _ { S } ( p ) \geq u _ { A }$ , that is, if $\Delta u ( p ) \leq 0$ . We resolve ties in favor of solving. Rearranging,

$$
\lambda + \gamma ( P _ { A I } - p ) \leq 0 \quad \Longleftrightarrow \quad \gamma p \geq \gamma P _ { A I } + \lambda \quad \Longleftrightarrow \quad p \geq P _ { A I } + \frac { \lambda } { \gamma } ,
$$

which is Eq. (5). Appropriate reliance solves if $p \geq P _ { A I }$ , so the myopic threshold lies higher by exactly $\lambda / \gamma$ . A worker with $P _ { A I } \leq p < P _ { A I } + \lambda / \gamma$ is more accurate than the AI but accepts. This is the effort-induced overreliance of Holstein et al. [33].

Running example. The myopic threshold is $. 6 2 + . 0 5 = . 6 7$ with the competent AI and . $. 7 7 + . 0 5 = . 8 2$ with the near-expert AI. Sam at .80 lies between .77 and .82 and therefore accepts, even though Sam is more accurate than the AI.

## A.4 The discount factor as a continuation probability

The lifetime value in $\operatorname { E q . }$ (6) weights the utility of task t by $\delta ^ { t }$ . Two descriptions of the worker lead to this weight. They differ in how many tasks the worker performs and in what the weight means, but not in the values or decisions they imply.

Discounting: the worker performs tasks indefinitely. In the first description, the worker performs an infinite sequence of tasks, and every task takes place with certainty. The worker, however, values a utility t tasks ahead at only $\delta ^ { t }$ times the same utility today. The weight $\delta ^ { t }$ is then a preference: it states how much the worker cares about a task that will certainly happen. The weights sum to

$$
\sum _ { t = 0 } ^ { \infty } \delta ^ { t } = \frac { 1 } { 1 - \delta } = H .
$$

An infinite stream of tasks with the same expected utility u is therefore worth $H u .$ , as much as $H$ undiscounted tasks. Under this description, H is an effective horizon. It measures how far ahead the worker looks. It is not the number of tasks the worker performs, because that number is infinite.

Continuation: the worker performs an expected number of tasks. In the second description, the worker does not discount. Every task that takes place counts fully. However, after each task another task of the same family follows only with probability δ, independently of the worker’s skill and choices. The task family itself continues indefinitely, but the worker’s participation in it ends at a random point, for example when the worker moves to another role. The weight $\delta ^ { t }$ is then a probability: it states how likely it is that a task the worker fully cares about takes place.

Let $T$ be the number of tasks the worker performs, including the current one. Task t (counting from $t = 0 )$ takes place if and only if the t continuations before it all happen, so $\operatorname* { P r } ( T > t ) = \delta ^ { t }$ . The expected total utility is

$$
\mathbb { E } \Big [ \sum _ { t = 0 } ^ { T - 1 } U _ { t } \Big ] = \sum _ { t = 0 } ^ { \infty } \operatorname* { P r } ( T > t ) \mathbb { E } [ U _ { t } ] = \sum _ { t = 0 } ^ { \infty } \delta ^ { t } \mathbb { E } [ U _ { t } ] .
$$

The first equality writes the random sum as a sum over all $t ,$ each counted only if task t takes place, and uses that the continuation is independent of $U _ { t }$ . The expected number of tasks is

$$
\mathbb { E } [ T ] = \sum _ { t = 0 } ^ { \infty } \operatorname* { P r } ( T > t ) = \sum _ { t = 0 } ^ { \infty } \delta ^ { t } = \frac { 1 } { 1 - \delta } = H .
$$

Under this description, H is the expected horizon, the average number of tasks the worker actually performs.

Why the two descriptions are equivalent. In both descriptions, task t enters the objective with weight $\delta ^ { t }$ . Under discounting, a certain task receives the reduced weight $\delta ^ { t } ;$ under continuation, a task with full weight takes place with probability $\delta ^ { t }$ . Because the worker maximizes an expectation, only the product of the two matters. Both descriptions therefore yield the objective in $\operatorname { E q . } \left( 6 \right)$ , the same lifetime values, and the same optimal decisions. This equivalence concerns the worker’s optimization problem; it does not imply equal training-completion probabilities or equal organizational payments over an actual career. The same sum $1 / ( 1 - \delta )$ is the effective horizon under discounting and the expected number of tasks under continuation. The main text uses the continuation reading because it gives $\delta$ a concrete meaning: how many more tasks of this family the worker expects to perform. Accordingly, H on the vertical axis of Figure 1 is the expected horizon. The myopic worker is the case $\delta = 0$ in both readings. Under continuation, the current task is then certainly the last one, so $\dot { H } = 1$

Expected versus fixed horizon. An expected horizon of H tasks differs from a fixed horizon of exactly H tasks. First, the realized number of tasks varies widely. T follows a geometric distribution with $\Pr ( T > m ) = \delta ^ { m }$ for every integer $m \geq 0$ . The worker therefore performs more than H tasks only with probability $\delta ^ { \lfloor H \rfloor }$ , which is close to $e ^ { - 1 } \approx . 3 7$ for large H. Second, the expected horizon does not shrink as tasks pass. A worker who has already performed t tasks still expects $\mathbb { E } [ T - t \mid T > { \dot { t } } ] = H$ tasks, including the current one, because each continuation has the same probability $\delta .$ Every task therefore looks the same to the worker, and the lifetime value depends only on the current skill level $p ,$ not on the task index t. The Bellman equation in Appendix A.5 relies on this property. With a fixed horizon, the value would also depend on the number of remaining tasks. On the final task, the decision reduces to the myopic rule: solving may still be worthwhile, but future skill provides no additional benefit.

Finite lifetime value. The lifetime value is always finite. The utility of every task lies between −λ and $\gamma ,$ so the sum in Eq. (6) is bounded in absolute value by $( \gamma + \lambda ) \bar { \sum _ { t } } \delta ^ { t } = ( \gamma + \lambda ) \bar { / } ( 1 - \delta )$

Running example. With $\delta = . 9 9 , H = 1 / ( 1 - . 9 9 ) = 1 0 0$ . Under discounting, Anna performs code reviews indefinitely and values a bugfound 100 tasksfrom now at $. 9 9 ^ { 1 0 0 } \approx . 3 7 $ ofa bugfound today. Under continuation, Anna values both equally but expects to perform 100 reviews, and Anna is still reviewing after 100 tasks with probability $9 9 ^ { 1 0 0 } \approx . 3 7 $ Both readings give the same weight. That H equals the $1 / \phi _ { d } = 1 0 0$ tasks until a decay step is a coincidence of the chosen parameters.

## A.5 The Bellman equation and the strategic decision rule

Why a Bellman equation. A general policy can specify actions for every possible history. In this finite discounted Markov decision process, however, an optimal deterministic stationary policy exists, so it suffices to consider the $2 ^ { N }$ state-based policies used here. Bellman reasoning avoids enumerating all of them. The Bellman equation replaces this search with a comparison of two actions on the current task [47]. Whatever the worker does now, the remaining tasks are best handled optimally, so their value is the lifetime value at the skill level the current action leads to. This argument requires that the remaining problem is the same at every task, which holds because the expected horizon does not shrink (Appendix A.4). The Bellman equation serves two purposes in the paper. First, it turns the strategic choice into a decision rule for a single task, Eq. (8), which can be compared directly with the myopic rule and with appropriate reliance. Second, it reduces the model to N equations that we solve exactly (Appendix A.6).

Bellman equation, Eq. (7). The Bellman equation splits the lifetime value into the current task and everything that follows. Suppose the worker at skill p accepts now and then acts optimally. The current task yields $u _ { A }$ in expectation. The next task takes place with weight δ and starts from skill $p ^ { - }$ <sup>−</sup> with probability $\phi _ { d }$ and from $p$ otherwise. From whichever level skill reaches, the remaining problem looks exactly like the original one: the same tasks, the same AI, and the same $\delta ,$ only from a possibly different level. Its best achievable value is therefore $V$ at that level. Accepting now thus yields

$$
Q _ { A } ( p ) = u _ { A } + \delta \bigl [ ( 1 - \phi _ { d } ) V ( p ) + \phi _ { d } V ( p ^ { - } ) \bigr ] ,
$$

and by the same argument solving now yields

$$
Q _ { S } ( p ) = u _ { S } ( p ) + \delta \bigl [ ( 1 - \phi _ { u } ) V ( p ) + \phi _ { u } V ( p ^ { + } ) \bigr ] .
$$

The worker chooses the better of the two, so $V ( p ) = \operatorname* { m a x } \{ Q _ { A } ( p ) , Q _ { S } ( p ) \}$ , which is Eq. (7). At the lowest level, $p ^ { - } = p .$ , so the bracket of $Q _ { A }$ equals $V ( p )$ ; at the highest level, $p ^ { + } = p ,$ and the same holds for $Q _ { S }$ . Because $V$ also appears on the right-hand side, Eq. (7) is a system of $N$ equations in the $N$ unknowns $V ( a _ { 1 } ) , \ldots , V ( a _ { N } )$ Appendix A.6 explains how we solve it.

Strategic decision rule, Eq. (8). The strategic worker solves if $Q _ { S } ( p ) \geq Q _ { A } ( p )$ . Writing out both sides,

$$
u _ { S } ( p ) + \delta \bigl [ ( 1 - \phi _ { u } ) V ( p ) + \phi _ { u } V ( p ^ { + } ) \bigr ] \geq u _ { A } + \delta \bigl [ ( 1 - \phi _ { d } ) V ( p ) + \phi _ { d } V ( p ^ { - } ) \bigr ] .
$$

Moving $u _ { S } ( p )$ to the right and the discounted terms of accepting to the left, the left-hand side becomes $u _ { A } - u _ { S } ( p ) =$ $\Delta u ( p )$ and the right-hand side becomes

$$
\delta \bigl [ ( 1 - \phi _ { u } ) V ( p ) + \phi _ { u } V ( p ^ { + } ) - ( 1 - \phi _ { d } ) V ( p ) - \phi _ { d } V ( p ^ { - } ) \bigr ] = \delta \bigl [ \phi _ { u } \bigl ( V ( p ^ { + } ) - V ( p ) \bigr ) + \phi _ { d } \bigl ( V ( p ) - V ( p ^ { - } ) \bigr ) \bigr ] ,
$$

which is Eq. (8). The step is the same as in the derivation of Eq. (2), with lifetime value V in place of next-task accuracy. The first term is the value of a possible learning step, and the second is the value of an avoided decay step. With $\delta = 0$ the right-hand side is zero and the rule reduces to $\bar { \Delta } u ( p ) \leq 0$ , the myopic rule.

Both terms are nonnegative whenever higher skill is worth at least as much over the remaining tasks, $V ( a _ { 1 } ) \leq \cdots \leq$ $V { \left( a _ { N } \right) }$ . Without incentives, Eq. (18) below shows this directly. The future value of independent work can therefore only make a strategic worker solve more often than a myopic one, never less.

Running example. With the competent AI and $\delta = . 9 9$ , the lifetime values relative to always accepting are 0, 0, 4.06, 9.64, and 13.00 at thefive levels (Appendix A.7). At .60, the learning term is $. 9 9 \times . 0 2 \times ( 9 . \dot { 6 } 4 - 4 . 0 6 ) = . 1 1 1$ and the retention term is $. 9 9 \times . 0 1 \times ( 4 . 0 \dot { 6 } - 0 ) = . 0 4 0$ . Their sum, .151, exceeds $\Delta u ( . 6 0 ) = . 0 7 ,$ , so Anna solves at .60. At .50, the learning term $i s . 9 9 \times . 0 2 \times ( 4 . 0 6 - 0 ) = . 0 8 0$ , and the retention term is zero. This is less than $\Delta u ( . 5 0 ) = . 1 7 ,$ so Anna accepts at .50.

## A.6 Solving the model

Transition matrices. For a fixed action at every level, the transition rule of Appendix A.1 is a matrix whose entry in row n and column m is the probability of moving from $a _ { n }$ to $a _ { m }$ . Solving at every level gives $T _ { S }$ , with $1 - \phi _ { u }$ on the diagonal and $\phi _ { u }$ just to the right of it. Accepting at every level gives $T _ { A } .$ , with $1 - \phi _ { d }$ on the diagonal and $\phi _ { d }$ just to the left of it. In the first row of $T _ { A }$ and the last row of ${ \dot { T _ { S } } } ,$ the step probability is added to the diagonal, because the worker stays at the boundary. A policy π picks one action per level. Its transition matrix $T _ { \pi }$ takes row n from $T _ { S }$ or $T _ { A }$ according to the action at $a _ { n } .$ , and $u _ { \pi }$ is the vector of the corresponding expected utilities $u _ { S } ( a _ { n } ) \ \mathrm { o r } \ u _ { A }$

Value of a fixed policy. For a fixed policy, the maximum in Eq. (7) disappears, and the equation becomes linear, $V _ { \pi } = u _ { \pi } + \delta T _ { \pi } \bar { V _ { \pi } }$ . Its solution is

$$
V _ { \pi } = ( I - \delta T _ { \pi } ) ^ { - 1 } u _ { \pi } ,\tag{15}
$$

where I is the identity matrix. The inverse exists because $\delta < 1$ and the rows of $T _ { \pi }$ sum to one. At a single level, if the worker solves at $a _ { n } < a _ { N }$ , then $V _ { \pi } ( a _ { n } ) = u _ { S } ( a _ { n } ) + \delta [ ( 1 - \phi _ { u } ) V _ { \pi } ( a _ { n } ) + \phi _ { u } V _ { \pi } ( a _ { n + 1 } ) ]$ , so

$$
V _ { \pi } ( a _ { n } ) = \frac { u _ { S } ( a _ { n } ) + \delta \phi _ { u } V _ { \pi } ( a _ { n + 1 } ) } { 1 - \delta + \delta \phi _ { u } } .
$$

Policy iteration. We find the optimal policy by policy iteration [47]. We start from the myopic policy, compute its value with Eq. (15), and then switch the action at every level at which the other action has a strictly higher value $Q _ { A }$ or $Q _ { S }$ . We repeat these two steps until no level switches. Each round weakly improves the value at every level. With only $2 ^ { \overset { \cdot } { N } }$ possible policies, the procedure stops after finitely many rounds (in practice, after two or three), and the final policy satisfies Eq. (7). Ties are resolved in favor of solving. Value iteration, which applies the right-hand side of Eq. (7) repeatedly, reaches the same solution but approaches it only at rate δ per round, which is slow for $\delta = . 9 9$ . Policy iteration is exact up to floating-point precision.

Expected trajectories. The figures show exact expectations over time rather than averages of simulated workers. Let $\mu _ { t }$ be the row vector of probabilities that the worker is at each level before task t, with $\mu _ { 0 }$ placing all mass on the starting level. Under a fixed policy, $\mu _ { t + 1 } = \mu _ { t } T _ { \pi }$ . Expected unaided accuracy before task t is $\bar { \sum _ { n } \mu _ { t } ( n ) } a _ { n }$ . Expected team accuracy, the accuracy of the submitted answer, replaces $a _ { n }$ by $P _ { A I }$ at levels where the worker accepts. Expected utility and expected incentive payments are computed in the same way from the per-level values. Under the continuation interpretation, these trajectories are conditional on the worker still participating at task t. Undiscounted organizational costs are reported separately as benchmarks for continued participation over an unlimited task sequence.

## A.7 A closed form for the strategic worker

Without incentives, the strategic worker’s choice has a simple closed form. It explains the shape of Figure 1 and the running-example values above.

Expected future accuracy. Consider a worker who starts at level $a _ { n }$ and solves on every task. Skill then only rises. Define the worker’s expectedfuture accuracy as the weighted average of unaided accuracy over all future tasks, with task t weighted by $( 1 - \delta ) \delta ^ { t }$

$$
\hat { a } _ { n } = ( 1 - \delta ) \sum _ { t = 0 } ^ { \infty } \delta ^ { t } \mathbb { E } \big [ p _ { t } \big | p _ { 0 } = a _ { n } , \mathrm { ~ s o l v e ~ o n ~ e v e r y ~ t a s k } \big ] .
$$

The weights sum to one, so $\hat { a } _ { n }$ lies between $a _ { n }$ and $a _ { N }$ , and $\hat { a } _ { N } ~ = ~ a _ { N }$ . Conditioning on the first task gives $\hat { a } _ { n } = ( \bar { 1 } - \delta ) a _ { n } + \delta [ ( 1 - \phi _ { u } ) \hat { a } _ { n } + \phi _ { u } \hat { a } _ { n + 1 } ]$ , and solving for $\hat { a } _ { n }$ ,

$$
\hat { a } _ { n } = \left( 1 - \omega \right) a _ { n } + \omega \hat { a } _ { n + 1 } , \omega = \frac { \delta \phi _ { u } } { 1 - \delta + \delta \phi _ { u } } .\tag{16}
$$

The weight ω is the expected discount until the next learning step, $\mathbb { E } [ \delta ^ { K } ]$ for the geometric waiting time K of Eq. (14). Starting from $\hat { a } _ { N } = a _ { N }$ , Eq. (16) gives all other levels by working downward. The lifetime utility of solving on every task is $\textstyle \sum _ { t } \delta ^ { t } \mathbb { E } [ \gamma p _ { t } - \lambda ] = ( \gamma \hat { a } _ { n } - \lambda ) / ( 1 - \delta )$ , and that of accepting on every task is $\gamma P _ { A I } / ( 1 - \delta )$ .

Solving rule. The strategic worker’s choice turns out to be a comparison between exactly these two plans. Intuitively, the levels at which the worker solves are all levels from some threshold upward. A worker who solves stays among these levels, because solving never lowers skill, and therefore solves on every future task. A worker who accepts stays among the accepting levels, because accepting never raises skill, and therefore accepts on every future task. Comparing the two plans gives the strategic decision rule in closed form,

$$
\mathrm { s o l v e ~ a t } a _ { n } \quad \Longleftrightarrow \quad { \hat { a } } _ { n } \geq P _ { A I } + { \frac { \lambda } { \gamma } } ,\tag{17}
$$

and the lifetime value relative to always accepting,

$$
V ( a _ { n } ) - \frac { \gamma P _ { A I } } { 1 - \delta } = \frac { \gamma \left[ \hat { a } _ { n } - P _ { A I } - \lambda / \gamma \right] ^ { + } } { 1 - \delta } .\tag{18}
$$

The strategic worker thus applies the myopic rule (5) with expected future accuracy $\hat { a } _ { n }$ in place of current skill $p .$ . With $\delta = 0 , \hat { a } _ { n } = a _ { n }$ and the rule is the myopic one. As δ approaches one, $\hat { a } _ { n }$ approaches $a _ { N }$ . Eq. (18) increases in $n ,$ so higher skill is never worth less, as used in Appendix A.5.

Why the closed form is exact. We show that the values in Eq. (18) satisfy the Bellman equation (7) at every level, and that the rule (17) attains its maximum. This suffices because the Bellman equation has only one solution. Applying its right-hand side to two candidate value functions shrinks the largest difference between them by at least the factor $\delta < 1$ , so two different candidates cannot both remain unchanged [47]. A policy that attains the maximum at every level is then optimal.

We write $A = \gamma P _ { A I } / ( 1 - \delta )$ for the value of always accepting, $W _ { n } = ( \gamma \hat { a } _ { n } - \lambda ) / ( 1 - \delta )$ for the value of always solving from $a _ { n } .$ , and $\dot { V _ { n } } = \operatorname* { m a x } \{ A , W _ { n } \}$ for the candidate value. The argument uses three facts.

1. $\hat { a } _ { n }$ increases in $n . \mathrm { \bf ~ B y } \mathrm { E q . } ( \mathrm { 1 6 } ) , \hat { a } _ { n }$ is a weighted average of $a _ { n }$ and $\hat { a } _ { n + 1 }$ , and $a _ { n } < a _ { n + 1 } \leq { \hat { a } } _ { n + 1 }$ . The levels at which $W _ { n } \geq A$ , where Eq. (17) solves, are therefore all levels from some threshold upward, and $V _ { n }$ is nondecreasing in n.

2. W satisfies the solving branch, $W _ { n } = u _ { S } ( a _ { n } ) + \delta [ ( 1 - \phi _ { u } ) W _ { n } + \phi _ { u } W _ { n + 1 } ]$ , with $W _ { N + 1 } = W _ { N }$ . This follows from the recursion that defines $\hat { a } _ { n }$ after multiplying it by $\gamma _ { : }$ , subtracting $\lambda = ( 1 - \delta ) \lambda + \delta \lambda$ on both sides, and dividing by $1 - \delta$

3. A satisfies the accepting branch with itself, $A = u _ { A } + \delta A$

Levels at which the rule solves $( W _ { n } \geq A )$ . By fact 1, the level above also solves, so $V _ { n } = W _ { n }$ and $V _ { n + 1 } = W _ { n + 1 }$ , and by fact 2 the solving branch equals $V _ { n }$ . The accepting branch is at most $V _ { n }$

$$
u _ { A } + \delta \big [ ( 1 - \phi _ { d } ) V _ { n } + \phi _ { d } V _ { n - 1 } \big ] \ \leq \ u _ { A } + \delta V _ { n } \ \leq \ ( 1 - \delta ) V _ { n } + \delta V _ { n } = V _ { n } .
$$

The first step uses that V is nondecreasing, the second tha $u _ { A } = ( 1 - \delta ) A \leq ( 1 - \delta ) V _ { n }$ . Solving therefore attains the maximum.

Levels at which the rule accepts $( W _ { n } < A )$ . By fact 1, every lower level also accepts, so $V _ { n } = V _ { n - 1 } = A$ , and by fact 3 the accepting branch equals $V _ { n }$ . For the solving branch, note first that $u _ { S } ( a _ { n } ) < u _ { A }$ , because $a _ { n } \leq \hat { a } _ { n } < P _ { A I } + \lambda / \gamma$ If the level above also accepts, $V _ { n + 1 } = A$ , and the solving branch is $u _ { S } ( a _ { n } ) + \delta A < u _ { A } + \delta A = A$ . If the level above solves, $V _ { n + 1 } = W _ { n + 1 }$ , and subtracting the equation of fact 2 from the solving branch gives

$$
u _ { S } ( a _ { n } ) + \delta \bigl [ ( 1 - \phi _ { u } ) { \cal A } + \phi _ { u } W _ { n + 1 } \bigr ] = W _ { n } + \delta ( 1 - \phi _ { u } ) ( { \cal A } - W _ { n } ) < { \cal A } ,
$$

because $W _ { n } < A$ and $\delta ( 1 - \phi _ { u } ) < 1$ . Accepting therefore attains the maximum.

Hence the values of Eq. (18) solve the Bellman equation, and the rule (17) is optimal. We also confirmed Eqs. (17) and (18) against the exact solution of Eq. (7) for the running example and for every configuration of the robustness analysis in Appendix B.

Level-dependent step probabilities. The argument does not use that $\phi _ { u }$ and $\phi _ { d }$ are the same at every level. If the learning probability at $a _ { n }$ is $\phi _ { u , n }$ and the decay probability is $\phi _ { d , n }$ , Eq. (16) holds with the level-specific weight $\omega _ { n } = \delta \phi _ { u , n } / ( 1 - \delta + \delta \phi _ { u , n } )$ in place of $\omega ,$ , and facts 1–3 and both cases go through unchanged. The strategic rule (17) and the values (18) therefore also hold when learning slows down or speeds up as workers progress, and decay still does not affect the choice.

The decay probability $\phi _ { d }$ enters neither fact and therefore does not appear in Eq. (17). The retention term in $\operatorname { E q . } \left( 8 \right)$ is nonnegative. It is strictly positive when $\delta > 0$ and the current level has greater value than its lower neighbor. It is zero at the lowest grid level and can also vanish at a threshold tie. It does not, however, move the boundary between solving and accepting. Decay determines how fast skill falls at the levels where the worker accepts, not which levels these are.

Running example. With $\delta = . 9 9$ and $\phi _ { u } = . 0 2 , \omega = . 6 6 4 ,$ , and Eq. (16) gives $\hat { a } _ { n } = { . 5 5 9 , . 6 4 0 , . 7 1 1 , . 7 6 6 } ,$ , and .80 at the five levels. A novice at .40 thus decides as if their accuracy were .559. With the competent $A I ,$ the threshold is .67, so the strategic worker solves at .60, .70, and .80 and accepts at .40 and .50. Eq. (18) gives the values $( . 7 1 1 - . 6 7 ) / . 0 1 = \bar { 4 } . 0 6 , 9 . 6 4$ , and 13.00 at the three solving levels. With the near-expert AI, the threshold is .82, above every $\hat { a } _ { n } ,$ , so the strategic worker accepts everywhere.

Policy regions in Figure 1. For a fixed starting level, Eq. (17) says that the strategic worker solves if and only if $P _ { A I } \leq \hat { a } _ { n } - \lambda / \gamma$ , where $\hat { a } _ { n }$ depends on the horizon through $\delta = 1 - 1 / H$ . This gives the boundary between the shaded regions. For the novice at .50, the boundary lies at $P _ { A I } \stackrel { = } { = } . 4 5$ for $H = 1$ , at $\bar { P _ { A I } } = . 5 9$ for $H = \mathrm { 1 0 0 }$ , and approaches $a _ { N } - \lambda / \gamma = . 7 5$ as H grows. With the competent $\mathrm { A I } \left( . 6 2 > . 5 9 \right)$ , Anna therefore still accepts at $H = 1 0 0$ . For the expert at $a _ { N } = . 8 0 , \hat { a } _ { N } = a _ { N }$ at every horizon, so the boundary is the vertical line $P _ { A I } = . 7 5$

## A.8 Low-skill trap and expert deskilling

Low-skill trap. Suppose the worker accepts at the two lowest levels $a _ { 1 }$ and $a _ { 2 }$ . Starting from either level, skill can only stay the same or fall, so the worker never leaves these two levels, accepts on every task, and earns $u _ { A }$ per task. Both levels therefore have the same value, $V ( a _ { 1 } ) = V ( a _ { 2 } ) = \gamma P _ { A I } / ( 1 - \dot { \delta } )$ . At $a _ { 1 }$ , the learning term in $\operatorname { E q . } \left( 8 \right)$ is $\delta \phi _ { u } [ V ( a _ { 2 } ) - V ( a _ { 1 } ) ] = 0$ , and the retention term is zero because $a _ { 1 } ^ { - } = a _ { 1 } . \mathrm { ~ A ~ }$ single learning step is worth nothing to a worker who would still accept one level higher. By Eq. (17), this situation arises exactly when $\hat { a } _ { 2 } < P _ { A I } + \lambda / \gamma$

Running example. With the competent AI, $\hat { a } _ { 2 } = . 6 4 0 < . 6 7 ,$ , so Anna accepts at .40 and .50, and $V ( . 4 0 ) = V ( . 5 0 )$ Starting at .50, expected skill after t tasks is . $. 4 0 + . 1 0 \times ( . 9 9 ) ^ { t }$ , whichfalls toward .40, while output accuracy stays at .62, because every submitted answer is the AI’s.

Expert deskilling. Suppose $a _ { N } < P _ { A I } + \lambda / \gamma$ . Then $\Delta u ( a _ { N } ) > 0$ , and because $\Delta u$ decreases in skill, $\Delta u ( p ) > 0$ at every level: at every level and on every task, accepting pays strictly more today than solving. No sequence of choices can therefore earn more than $u _ { A }$ on any task, and always accepting earns exactly $u _ { A }$ on every task. Hence

$$
V ( p ) = \sum _ { t = 0 } ^ { \infty } \delta ^ { t } \gamma P _ { A I } = \frac { \gamma P _ { A I } } { 1 - \delta } \qquad \mathrm { f o r } \mathrm { e v e r y } p \in \mathcal { P } .
$$

Since V is the same at every level, both differences in Eq. (8) are zero, and the worker accepts everywhere for any $\delta .$ While the worker always accepts, let $B _ { t }$ count potential decay events over t tasks:

$$
B _ { t } \sim \operatorname { B i n o m i a l } ( t , \phi _ { d } ) .
$$

Starting at $a _ { N }$ , the actual number of downward moves is capped by the lower boundary:

$$
D _ { t } = \mathrm { m i n } \{ B _ { t } , N - 1 \} , \qquad p _ { t } = a _ { N - D _ { t } } .
$$

For equally spaced levels with distance $s ,$

$$
\mathbb { E } [ p _ { t } ] = a _ { N } - s \mathbb { E } [ \operatorname* { m i n } \{ B _ { t } , N - 1 \} ] .
$$

Before the lower boundary can affect the process, expected skill therefore falls by $\phi _ { d } s$ per task. For $N \geq 2$ , the probability of remaining at the highest level is $( 1 - \phi _ { d } ) ^ { t }$ . For $0 < \phi _ { d } < 1$ , its half-life is ln $2 / [ - \ln ( 1 - \phi _ { d } ) ]$ approximately $. 6 9 / \phi _ { d }$ for small $\phi _ { d } . \mathrm { ~ H ~ } \phi _ { d } = 1$ , the worker leaves the highest level after the first task.

Running example. With the near-expert AI, $. 8 0 < . 7 7 + . 0 5 = . 8 2 ,$ , so Sam accepts at every level. Sam is still at .80 with probability one halfafter 69 tasks. Expected skillfalls by . $. 0 1 \times . 1 0 = . 0 0 1$ per task atfirst and reaches .70 after about 100 tasks, while output accuracy stays at .77.

## A.9 Incentives

Incentive compatibility. Under the incentive, a correct independent answer earns $\gamma + \theta ( p )$ instead of $\gamma .$ . Taking expectations with $\operatorname* { P r } ( Y _ { t } = 1 ) = p \operatorname { g i v e s } { ( \gamma + \theta ( p ) ) p - \lambda } = u _ { S } ( p ) + p \theta ( p )$ for solving, while accepting still gives $u _ { A }$ The myopic worker solves if

$$
u _ { S } ( p ) + p \theta ( p ) \geq u _ { A } \quad \Longleftrightarrow \quad p \theta ( p ) \geq u _ { A } - u _ { S } ( p ) = \Delta u ( p ) ,
$$

which is Eq. (9). The product $p \theta ( p )$ is the incentive the worker expects per solved task, and it is also what the organization expects to pay per solved task.

Effort incentive, Eq. (10). With $\theta ^ { \mathrm { e f f } } ( p ) = \lambda / p .$ the expected payment per solved task is $p \cdot \lambda / p = \lambda , \mathrm { s o } u _ { S } ^ { \mathrm { e f f } } ( p ) =$ $\gamma p - \lambda + \lambda = \gamma p$ . The myopic worker solves if $\gamma p \geq \gamma P _ { A I }$ , that is, if $p \geq P _ { A I }$ , which is appropriate reliance. The effort incentive acts as if solving cost no effort. Repeating the argument for expert deskilling in Appendix A.8 with $\lambda = 0$ shows that the expert accepts at every level once $a _ { N } < P _ { A I }$ , so the effort incentive raises the AI accuracy at which experts start to deskill from $a _ { N } - \lambda / \overset { \cdot } { \gamma }$ to $a _ { N }$ . For a strategic worker, Eq. (17) with $\lambda = 0$ gives solving if and only if $\hat { a } _ { n } \geq P _ { A I }$

Running example. Under the effort incentive, Sam at .80 solves with the near-expert AI $( . 8 0 \geq . 7 7 )$ and receives .05 per task in expectation. With the above-expert AI, $. 8 0 < . 8 2$ , so Sam accepts at every level. Anna’s levels .40, .50, and .60 all lie below .62, so a myopic Anna accepts at each of them.

Learning incentive, Eqs. (11) and (12). Eq. (9) requires $\theta ( p ) \geq \Delta u ( p ) / p . { \mathrm { ~ I f ~ } } \Delta u ( p ) \leq 0$ , the worker solves without any incentive, and the smallest nonnegative incentive is zero. Otherwise it is $\Delta u ( p ) / p$ . Both cases together give $\theta ^ { \mathrm { m i n } } ( p ) = [ \Delta u ( p ) ] ^ { + } / p$ . Where $\Delta u ( p ) \overset { \textstyle \cdot } { \geq } 0$ , splitting the numerator $\Delta u ( p ) = \lambda + \gamma ( \ddot { P _ { A I } } - p )$ and dividing each part by p gives Eq. (12). The expected payment per solved task is $p \theta ^ { \mathrm { m i n } } ( p ) \stackrel { \sim } { = } [ \Delta u ( p ) ] ^ { + }$ , the positive part of the immediate payoff gap. Under the learning incentive, the expected utility of solving becomes $\bar { u _ { S } } ( p ) + [ \bar { \Delta } u ( p ) ] ^ { + } = \operatorname* { m a x } \{ u _ { A } , u _ { S } ( p ) \}$ so it matches accepting where accepting would pay more and leaves the worker’s payoff unchanged otherwise.

Running example. Anna at .60 with the competent AI receives $\theta ^ { \mathrm { m i n } } = . 0 7 / . 6 0 = . 1 1 7$ per correct independent answer, or .07 per task in expectation, ofwhich .05 isfor effort and .02for the accuracy gap. Sam at .80 with the near-expert AI receives .02 per task in expectation, and .07 with the above-expert AI.

From myopic to strategic workers. Under an incentive, the solving branch of the Bellman equation (7) gains the expected payment $p \theta ( p )$ , and the strategic worker solves if $p \theta ( p ) + F ( \bar { p } ) \geq \Delta u ( p )$ , where $F ( p )$ denotes the right-hand side of Eq. (8) computed with the lifetime values under the incentive. Where the incentive moves a myopic worker, $p \theta ( p ) \geq \Delta u ( p )$ already holds, so the strategic worker also solves as long as $F ( p ) \geq 0$ , that is, as long as higher skill is worth at least as much under the incentive. This holds for the incentives in Section 5 in the running example and in every configuration of the robustness analysis (Appendix B).

Target incentive, Eq. (13). The target incentive is the learning incentive paid only at levels $p \leq p ^ { \star }$ . At and below the target, the myopic worker therefore solves. Above it, the worker faces the unpaid choice and solves only if $\Delta u ( p ) \leq 0$

Cost of acquiring a target. Organizational costs in Section 5 are expected payments to a myopic worker, summed over an unlimited sequence of tasks without discounting. These are not expected payments over a geometrically terminated career. Suppose the worker starts at $a _ { i }$ with $i \overset { \cdot } { \le } j < N$ , the target is $p ^ { \star } = a _ { j } ,$ , and solving pays without an incentive one level above it, $\Delta u ( a _ { j + 1 } ) \leq 0$ . Because $\Delta u$ decreases in skill, solving then pays on its own at every level above the target as well. The worker solves at every level from $a _ { i }$ upward, so skill never falls. At each supported level $a _ { n } .$ the worker spends on average $1 / \phi _ { u }$ tasks (Eq. (14)) and receives $[ \Delta u ( a _ { n } ) ] ^ { + }$ per task in expectation. Whether a learning step happens does not depend on whether the answer was correct, so the expected payment while at $a _ { n }$ is the product of the two. Summing over the supported levels,

$$
\mathrm { e x p e c t e d \ t o t a l \ p a y m e n t } = \frac { 1 } { \phi _ { u } } \sum _ { n = i } ^ { j } \left[ \Delta u ( a _ { n } ) \right] ^ { + } .\tag{19}
$$

After the worker reaches $a _ { j + 1 }$ , payments stop permanently. If instead $\Delta u ( a _ { j + 1 } ) > 0$ , the worker accepts just above the target, eventually decays back to it, and the incentive is paid again on every return, so payments recur indefinitely.

For the top target $j = N$ , payments end if $\Delta u ( a _ { N } ) \leq 0$ . In this case, the expected total payment is the sum of the transient-state costs,

$$
\frac { 1 } { \phi _ { u } } \sum _ { n = i } ^ { N - 1 } [ \Delta u ( a _ { n } ) ] ^ { + } ,
$$

with an empty sum equal to zero. If $\Delta u ( a _ { N } ) > 0 .$ , the top state requires a positive expected payment on every task. A worker already in an unpaid invariant solving region requires no payments, irrespective of a lower target.

Running example. With the competent AI, starting at .40 with target .60, solving pays on its own at .70 $( \Delta u ( . 7 0 ) =$ $- . 0 3 ) _ { ; }$ , and Eq. (19) gives $5 0 \times ( . 2 7 + . 1 7 + . 0 7 ) = 2 5 . 5 0$ . Starting at .50 with the learning incentive at every level, as in Figure 2A, it gives $5 0 \times ( . 1 7 + . 0 7 ) = 1 2 . 0 0$ . With target .50, Anna accepts at .60 and payments recur: each return to .50 costs $5 0 \times . 1 7 = 8 . 5 0$ and is followed by 100 unpaid tasks at .60 on average, a long-run rate of $8 . 5 0 / 1 5 0 \approx . 0 5 7$ per task. The same counting gives the additional errors ofpractice in Section 6.4. While practicing at .40, .50, and .60 instead ofaccepting, Anna submits $5 0 \times [ ( . 6 2 - . 4 0 ) + ( . 6 2 - . 5 0 ) + ( . 6 2 - . 6 0 ) ] = 1 8$ more incorrect answers in expectation than the AI would.

Cost of retaining a target. At the target, the worker solves, so skill cannot fall below it. Above the target, accepting lowers skill by at most one level per task, so skill must pass through the target before it could fall further, and there the worker solves again. If the target is the highest level, $p ^ { \star } = a _ { N }$ , the worker solves on every task and the organization pays $[ \Delta u ( a _ { N } ) ] ^ { + }$ per task for as long as the skill is to be retained.

If the target is one level lower, $p ^ { \star } = a _ { N - 1 }$ , and the worker accepts at $a _ { N } \left( \Delta u ( a _ { N } ) > 0 \right)$ , skill alternates between the two levels. Each cycle consists of on average $1 / \phi _ { d }$ unpaid tasks at $a _ { N }$ and $1 / \phi _ { u }$ supported tasks at $a _ { N - 1 }$ , each paying $\Delta u ( a _ { N - 1 } )$ in expectation. The long-run payment per task is the expected payment per cycle divided by the expected length of a cycle,

$$
\frac { \Delta u ( a _ { N - 1 } ) / \phi _ { u } } { 1 / \phi _ { u } + 1 / \phi _ { d } } = \Delta u ( a _ { N - 1 } ) \frac { \phi _ { d } } { \phi _ { u } + \phi _ { d } } .
$$

The higher target is therefore cheaper in the long run if $\Delta u ( a _ { N } ) ( \phi _ { u } + \phi _ { d } ) < \Delta u ( a _ { N - 1 } ) \phi _ { d }$ . With equally spaced levels at distance $s , \Delta u ( a _ { N - 1 } ) = \bar { \Delta } u ( a _ { N } ) + \gamma s$ , and the condition simplifies to $\Delta u ( a _ { N } ) \phi _ { u } < \gamma s \phi _ { d }$ . This condition holds when the expert’s remaining payoff gap is small relative to the larger gap one level down.

Running example. With the near-expert AI, target .80 costs $\Delta u ( . 8 0 ) = . 0 2$ per task. Target .70 costs $\Delta u ( . 7 0 ) = . 1 2$ per supported task, and a cycle consists of 100 unpaid and 50 supported tasks on average, so the long-run payment is $. 1 2 \times 5 0 / 1 5 0 = . 0 4$ per task. The condition reads . $. 0 2 \times . 0 2 = . { \bar { 0 0 0 4 } } < . 1 0 \times . 0 1 = . 0 { \bar { 0 1 } }$

## A.10 Strategic return of independent work

Figure 4 traces the running example of Section 4.2 over time. Starting at $p _ { 0 } = . 6 0$ with the competent AI, the strategic worker solves and initially earns less than a myopic worker, who accepts. For both policies we compute the expected utility of each task as in Appendix A.6 and plot the cumulative difference without discounting. The strategic worker gives up $\Delta u ( . 6 0 ) = . 0 7$ per task at first. As skill rises, the shortfall per task shrinks and turns into a gain. The cumulative shortfall is largest after 38 tasks and is offset after 80 tasks. Thereafter the gain grows by up to $\bar { . 8 0 } - . 0 5 - . 6 2 = . 1 3$ per task, the advantage of an expert who solves over a worker who accepts. Weighting task t by $\delta ^ { t }$ instead gives the discounted lifetime gain of 4.06, which equals the closed-form value $( . 7 \bar { 1 } 1 - . 6 7 \bar { ) } / . 0 1$ of Eq. (18), because a myopic worker at .60 accepts on every task.

![](images/ac4a23e18bef227ea209a4f542df2101d8d257147a60e24a35aabbdaae16bcec.jpg)  
Figure 4: Strategic independent work pays off despite an initial cost. Starting at $p _ { 0 } = . 6 0$ with the competent AI $( P _ { A I } = . 6 2 )$ , the curve shows expected cumulative undiscounted utility under the strategic policy minus that under the myopic policy. The largest shortfall is 1.30γ after 38 tasks, and the gain first becomes nonnegative after 80 completed tasks. The discounted lifetime gain is 4.06γ. Parameters as in Section 3.2, with $\delta = . 9 9$

## B Robustness analysis

Sections 4 and 5 illustrate the results with one set of parameters. This appendix shows that the qualitative results do not depend on that choice. We vary every parameter of the model over a wide range, solve the model exactly for every combination, and check each qualitative result (Appendix B.2). Because the decisions depend on the parameters only through a few combinations, the results can also be summarized in two figures (Appendix B.3). Finally, we vary the running example one parameter at a time to show which of its specific outcomes depend on the chosen values (Appendix B.4).

## B.1 Design

Table 4 lists the parameter values. We combine all of them in a full factorial design. The reward is fixed at $\gamma = 1$ because decisions depend on λ and $\gamma$ only through $\lambda / \gamma$ (Appendix A.2). The eight skill grids vary the number of levels, the range of accuracies, and the spacing between levels. With a constant step probability, unequal spacing makes the accuracy gained per learning step change with skill, which stands in for learning that slows down or speeds up as workers progress. The learning and decay probabilities are stated per step of .10, as in the running example. On grids with a different average step, we multiply the probabilities by .10 divided by the mean step size. This preserves interior expected accuracy drift on uniform grids. On unequal grids, local drift still depends on the size of the adjacent step. A grid with steps of .05 thus uses $2 \phi _ { u }$ and $2 \phi _ { d }$ . Combinations in which a scaled probability would exceed one are dropped.

Table 4: Parameter values in the full factorial design. Running-example values in bold.
<table><tr><td>Parameter</td><td>Values</td><td>Count</td></tr><tr><td>Effort cost  $\lambda / \gamma$ </td><td>.01, .025, .05, .10, .20</td><td>5</td></tr><tr><td>Learning probability  $\phi _ { u }$  (per step of .10)</td><td>.005, .01, .02, .05, .10, .25</td><td>6</td></tr><tr><td>Decay probability φd (per step of .10)</td><td>.0025, .005, .01, .02, .05, .10</td><td>6</td></tr><tr><td>Horizon  $H = 1 / ( 1 - \bar { \delta } )$ </td><td>2, 5, 10, 20, 50, 100, 200, 500, 1000</td><td>9</td></tr><tr><td>AI accuracy  $P _ { A I }$ </td><td>.30, .31, ..., .96</td><td>67</td></tr><tr><td>Skill grid</td><td>5 levels on [.40, .80]; 3, 9, or 17 equally spaced levels on [.40, .80]; 7 levels on [.30, .90]; 8 levels on [.60, .95]; diminishing steps {.40, .50, .59, .67, .74, .80}; increasing steps {.40, .44, .50, .58, .68, .80}</td><td>8</td></tr></table>

This gives 1,440 combinations of grid, effort cost, and step probabilities, each solved for 67 AI accuracies and 9 horizons, or 868,320 configurations in total. For each configuration, we compute the myopic policy and the strategic policy by exact policy iteration (Appendix A.6), without incentives, with the effort incentive, with the learning incentive at every level, and with a target incentive for every possible target. Incentive costs are numerical expectations computed from the transition matrices for myopic and strategic workers. For the sampled policies, long-run payment rates are computed by repeated squaring of the transition matrix, and total payments use expected transient-state visit counts. The policy comparison tolerance is $1 0 ^ { - 9 }$ ; the closed-form policy check excludes levels within $1 0 ^ { - 8 }$ of a threshold. For periodic chains permitted at the boundary of the model’s parameter domain, long-run payment rates are interpreted as time averages rather than pointwise limits. Ties are resolved in favor of solving, as in the main text.

## B.2 Which results hold in every configuration

Table 5 lists the checks. Every check holds in every configuration.

Strategic decisions. Rows 1–4 confirm the closed form of Appendix $\mathrm { A . 7 }$ across all configurations: the strategic worker solves exactly at the levels at which expected future accuracy $\hat { a } _ { n }$ is at least $P _ { A I } + \lambda / \gamma$ , and lifetime values follow Eq. (18). This closed form reduces the parameter space. Decay $\phi _ { d }$ does not enter it (row 7), and the learning probability and the horizon enter only through the weight ω of Eq. (16). With $\delta = 1 - 1 / H$

$$
\omega = \frac { \delta \phi _ { u } } { 1 - \delta + \delta \phi _ { u } } = \frac { \kappa } { 1 + \kappa } , \qquad \kappa = \phi _ { u } ( H - 1 ) .
$$

Since $H - 1$ is the expected number of tasks after the current one, κ is the expected number of learning steps a worker who always solves could take over the remaining tasks, ignoring the top level. A slow learner with a long horizon therefore decides exactly like a fast learner with a short one if both expect the same number of learning steps. Every strategic decision without incentives thus depends only on the skill grid, the effort cost $\lambda / \gamma$ , the AI accuracy $P _ { A I }$ , and $\kappa .$

Low-skill trap. For a worker who starts at level $p _ { 0 } < a _ { N }$ , the closed form yields three regions of AI accuracy. The worker solves at $p _ { 0 }$ if $P _ { A I } \leq \hat { a } ( p _ { 0 } ) - \lambda / \gamma$ . Everyone, including the expert, accepts if $P _ { A I } > a _ { N } - \lambda / \gamma$ . In between,

$$
\hat { a } ( p _ { 0 } ) - \lambda / \gamma < P _ { A I } \leq a _ { N } - \lambda / \gamma ,
$$

Table 5: Qualitative results checked in the full factorial design. Each row counts the configurations in which the condition applies and those in which it fails. Rows $1 { - } 7$ concern workers without incentives, rows 8–14 the incentives of Section 5. Rows 11 and 13 count every target in every configuration; rows 8, 12, and 14 do not depend on the horizon; row 7 counts combinations of the other parameters, each compared across the six decay probabilities.
<table><tr><td></td><td>Result</td><td>Checked</td><td>Failed</td></tr><tr><td>1</td><td>Strategic policy is a threshold in skill (solve at  $a _ { n } \Rightarrow$  solve at all  $a _ { m } > a _ { n } )$ </td><td>868,320</td><td>0</td></tr><tr><td>2</td><td>Strategic policy equals the closed form, Eq.  $( 1 7 ) ^ { a }$ </td><td>868,320</td><td>0</td></tr><tr><td>3</td><td>Lifetime values equal the closed form, Eq. (18)</td><td>868,320</td><td>0</td></tr><tr><td>4</td><td>V is nondecreasing in skill; the strategic worker solves wherever the myopic one does</td><td>868,320</td><td>0</td></tr><tr><td>5</td><td>Low-skill trap: if the worker accepts at  $a _ { 1 }$  and  $a _ { 2 } .$  then  $V ( a _ { 1 } ) = V ( \dot { a } _ { 2 } ) $ </td><td>520,524</td><td>0</td></tr><tr><td>6</td><td>Expert deskilling: all levels accept and  $V$  is constant iff  $a _ { N } < P _ { A I } + \lambda / \gamma$ </td><td>868,320</td><td>0</td></tr><tr><td>7</td><td>Strategic policy does not depend on  $\phi _ { d }$ </td><td>144,720</td><td>0</td></tr><tr><td>8</td><td>Effort incentive: myopic worker solves iff  $p \geq P _ { A I }$  (appropriate reliance)</td><td>96,480</td><td>0</td></tr><tr><td>9</td><td>Effort incentive: strategic worker solves iff  $\hat { a } _ { n } \geq P _ { A I } ;$  all accept iff  $a _ { N } < P _ { A I }$ </td><td>868,320</td><td>0</td></tr><tr><td>10</td><td>Learning incentive  $\theta ^ { \mathrm { m i n } }$  at every level moves the strategic worker to solve everywhere</td><td>868,320</td><td>0</td></tr><tr><td>11</td><td>Target incentive moves the strategic worker to solve at every level  $\leq p ^ { \star }$ </td><td>6,620,940</td><td>0</td></tr><tr><td>12</td><td>Myopic target policy: payment-ending conditions and acquisition costs or recurrent payment rates (App. A.9)</td><td>735,660</td><td>0</td></tr><tr><td>13</td><td>Strategic workers never cost more in the long run than myopic workers under the</td><td>6,620,940</td><td>0</td></tr><tr><td>14</td><td>same target Retention: long-run rates of targets  $a _ { N }$  and  $a _ { N - 1 }$  and the cost condition of App. A.9</td><td>29,772</td><td></td></tr></table>

$^ a \mathrm { U p }$ to numerical ties. In 6 configurations (17 levels, $\lambda / \gamma = . 0 2 5 , \phi _ { u } = . 1 0 , H = 2 , P _ { A I } = . 4 6 , \mathrm { a n y } \phi _ { d } ) , \hat { a } _ { n }$ falls short of $\bar { P _ { A I } ( + \lambda / \gamma }$ by less than $1 0 ^ { - 9 }$ at one level, and the two computations break this near-tie differently.

solving pays on its own at the top level, but the worker at $p _ { 0 }$ accepts and never gets there. Because AI accuracy is a probability, the feasible trap region is

$$
( \hat { a } ( p _ { 0 } ) - c , \ a _ { N } - c ] \cap [ 0 , 1 ] , \qquad c = \lambda / \gamma .
$$

Its width is

$$
L ( c ) = \left[ \operatorname* { m i n } \{ 1 , a _ { N } - c \} - \operatorname* { m a x } \{ 0 , \hat { a } ( p _ { 0 } ) - c \} \right] ^ { + } .
$$

When the interval is not clipped by the probability boundaries, this reduces to $a _ { N } - \hat { a } ( p _ { 0 } )$ and is independent of the effort cost and decay probability. This condition holds throughout the sampled design. For $p _ { 0 } < a _ { N }$ , the unclipped width is positive at every finite horizon, shrinks as κ grows, and tends to zero as the horizon becomes unbounded. Across the sampled grids, learning probabilities and horizons, its width at the lowest level ranges from .003 to .60. Outside this parameter range, clipping can make the feasible width depend on effort or eliminate the trap. Row 5 confirms that wherever the worker accepts at the two lowest levels, these levels have the same lifetime value.

Expert deskilling. Row 6 confirms that the expert accepts at every level and that lifetime value is the same at every level exactly when $a _ { N } < P _ { A I } + \lambda / \gamma$ , for every horizon, learning probability, decay probability, and grid. The boundary $a _ { N } - \lambda / \gamma$ is the vertical line in Figure 1B in every configuration.

Effort incentive. Row 8 confirms that the effort incentive turns the myopic worker’s choice into appropriate reliance. Row 9 extends Section 5.1 to strategic workers. Under the effort incentive, solving pays γp in expectation, so the closed form applies with $\lambda = 0 \colon$ the strategic worker solves at $a _ { n }$ if and only if $\hat { a } _ { n } \geq \bar { P _ { A I } } .$ , and all levels accept if and only if $a _ { N } < P _ { A I }$ . The effort incentive thus shifts both boundaries up by $\lambda / \gamma$ but leaves the unclipped width of the trap unchanged. A novice at $p _ { 0 }$ with foresight and paid effort still accepts whenever $\hat { a } ( p _ { 0 } ) < P _ { A I } \leq a _ { N }$ , although becoming an expert would pay. Foresight and the effort incentive together therefore do not close the low-skill trap either.

Learning and target incentives. Section 5 designs incentives for myopic workers and argues that they also move strategic workers (Appendix A.9). Rows 10 and 11 confirm this in every configuration. Under the learning incentive, the strategic worker solves at every level, and under a target incentive at every level up to the target. Row 13 shows that strategic workers never cost more in the long run than myopic workers under the same target. They solve at least as often, sometimes also just above the target, and can thereby reach levels at which payments end.

Acquiring a target. Row 12 checks the myopic target policy for every target $p ^ { \star } = a _ { j }$ , starting at $a _ { 1 }$ . For $j < N ,$ payments end if and only if $\Delta u ( a _ { j + 1 } ) \leq 0$ , and their expected total equals Eq. (19). For $\bar { \boldsymbol { \jmath } } = N$ , payments end if and only i $\mathrm { ~ f ~ } \Delta u ( a _ { N } ) \le 0$ . Otherwise, the long-run expected payment per task is

$$
\Delta u ( a _ { j } ) \frac { \phi _ { d } } { \phi _ { u } + \phi _ { d } } \quad \mathrm { f o r ~ } j < N , \qquad \Delta u ( a _ { N } ) \quad \mathrm { f o r ~ } j = N .
$$

For a myopic worker starting at $a _ { 1 }$ , whether payments end depends on the target’s position relative to the myopic threshold $\bar { P } _ { A I } + \lambda / \gamma$ . If this threshold lies above $a _ { 1 }$ and at or below $a _ { N }$ , the highest grid level below it is the smallest sufficient support cutoff. Larger targets induce the same bonus schedule and have the same cost, because the additional bonuses are zero. If $a _ { 1 } \geq \bar { P } _ { A I } + \bar { \lambda } / \gamma$ , no support is needed. If $a _ { N } < P _ { A I } + \lambda / \gamma$ , no target ends payments under continued participation.

Retaining a target. Row 14 confirms the long-run payments of the targets $a _ { N }$ and $a N { - } 1$ for a worker starting at a , and the condition $\Delta u ( a _ { N } ) ( \phi _ { u } + \phi _ { d } ) < \Delta \bar { u ( a _ { N - 1 } ) } \bar { \phi } _ { d }$ under which the higher target costs less. Unlike the other results, the comparison itself depends on the parameters. Among the 29,772 configurations with something to retain, $\Delta u ( a _ { N } ) > 0$ , the higher target costs less in 9,907 (33%). Figure 6 shows the boundary for the running-example grid. The higher target is cheaper when $P _ { A I }$ lies only slightly above the expert boundary $a _ { N } - \lambda / \gamma$ , so that $\Delta u ( a _ { N } )$ is small, and when skill decays quickly relative to how fast it is regained. For equally spaced levels with step s, the condition reads $\phi _ { u } / \phi _ { d } < \gamma s / \Delta u ( a _ { N } )$ . With the above-expert $\mathrm { A I } ( P _ { A I } = . 8 2 )$ , the running example lies on the other side of the boundary: $\gamma s / \Delta u ( . 8 0 ) = . 1 0 / . 0 7 = 1 . 4 3 < \phi _ { u } / \phi _ { d } = 2 ,$ so the lower target .70 costs less in the long run, at .06 instead of .07 per task, although it preserves less skill.

## B.3 Policy regions across parameters

## A Policy regions

![](images/419a5ebf155fe089c9b348191713c5f47d11cb7a3ce57d55eeb8126517c4382d.jpg)

## B Low-skill trap across skill grids

![](images/671dce435081ae4722d64ce6f55df2da1fd931631a551f5791df91d8a75b1b12.jpg)  
Figure 5: The low-skill trap and expert deskilling across parameters. (A) Strategic choice of a novice at $p _ { 0 } = . 5 0$ on the running-example grid with $\lambda / \bar { \gamma } = . 0 5$ , as a function of the expected number of learning steps over the horizon, $\kappa = \phi _ { u } ( H - 1 )$ , and the AI accuracy $P _ { A I }$ . Below the curve, the novice solves. Between the curve and $P _ { A I } = a _ { N } - \lambda / \gamma ,$ solving would pay at the top level, but the novice accepts (low-skill trap). Above that line, everyone accepts (expert deskilling). The curve is the closed form of Eq. (17); markers are boundaries computed with the exact solver for three learning probabilities and several horizons, which fall on the same curve. The running example has $\kappa = . 0 2 \times 9 9 = 1 . 9 8$ Decay $\phi _ { d }$ does not affect the regions. (B) Width between the theoretical trap boundaries, $a _ { N } - \hat { a } ( . 5 0 )$ , for six skill grids. The horizontal axis uses the baseline learning probability $\phi _ { u } ^ { ( . 1 0 ) }$ , with each grid’s probability scaled as described in Appendix B.1. The plotted width is measured before restricting the trap interval to feasible AI accuracies $P _ { A I } \in [ 0 , 1 ]$ If both boundaries lie within [0, 1], this equals the feasible trap width and is independent of $\lambda / \gamma$ . If a boundary lies outside [0, 1], only the intersection with [0, 1] is feasible; its width can be smaller and depend on $\lambda / \gamma$ . Neither width depends on the decay probabilit $\phi _ { d }$

Figure 5A summarizes the strategic choice of the novice for all learning probabilities, horizons, and decay probabilities at once. The running example lies in the middle of the range. With few expected learning steps, the boundary approaches the myopic one, $P _ { A I } = . 4 5$ , and the trap covers almost every AI that would be worth beating. With many steps, the trap narrows toward the line $P _ { A I } = a _ { N } - \lambda / \gamma$ , but it does not close. Anna is trapped with the competent AI because $\bar { P _ { A I } } = . 6 2 $ lies above the boundary at $\kappa = 1 . 9 8$ , which is .59. From about $\kappa = 2 . 9$ on, for instance with $H = 1 5 0$ instead of 100, the boundary passes .62 and Anna would solve, but the trap would then apply to more accurate AIs. Sam deskills with the near-expert AI at every κ.

Figure 5B shows similar trap widths across the examined discretizations of $[ . 4 0 , . 8 0 ]$ after normalizing step probabilities by mean grid spacing. Finer grids and grids with diminishing or increasing steps give almost the same width as the running example. The width grows with the distance between the starting level and the top level, as for the wider grid on [.30, .90], whose top level is higher. The low-skill trap is thus a matter of how far a novice is from the level at which independent work pays and of how many learning steps the remaining tasks allow, not of the particular grid.

![](images/a36ecfae59726c77432a15fe365b5b9b077688b09777b2794f2f29824e0c41b6.jpg)  
Figure 6: Which retention target costs less in the long run, for a worker at $a _ { N } = . 8 0$ on the running-example grid. The higher target $a _ { N }$ is cheaper below the boundary $\phi _ { u } / \phi _ { d } = \gamma s / \Delta u ( a _ { N } )$ , the lower target $a N _ { - 1 }$ above it. The solid boundary uses $\lambda / \gamma = . 0 5$ , the others .025 and .10. Dots are exact long-run payments fo $\lambda / \gamma = . 0 5$ , colored by the cheaper target. The two marked points are the running example with the near-expert AI $( \dot { P } _ { A I } = . 7 7 )$ ) and the above-expert AI $( P _ { A I } = . 8 2 )$ , both with $\phi _ { u } / \phi _ { d } = 2$

## B.4 The running example under one-at-a-time variation

The qualitative results hold in every configuration, but whether a particular worker falls into one of the regions depends on the parameters. Table 6 varies one parameter of the running example at a time and keeps $P _ { A I } = . 6 2$ for the competent and .77 for the near-expert AI.

Low-skill trap. Anna remains trapped under every variation except faster learning $( \phi _ { u } = . 0 5 )$ and longer horizons $( H \geq 2 0 0 )$ . In these cases the boundary rises above .62, because Anna expects enough learning steps for practice to pay. The trap width stays positive in every row, so the trap then applies to AIs between the new boundary and .75. Decay and the effort cost leave the width unchanged; the effort cost moves both boundaries by the same amount.

Expert deskilling. Sam deskills under every variation except the lower effort cost $\lambda / \gamma = . 0 2 5$ . Then $a _ { N } - \lambda / \gamma = . 7 7 5$ exceeds .77, Sam solves without an incentive, and nothing needs to be retained. No horizon, learning probability, decay probability, or grid on [.40, .80] changes Sam’s choice.

Incentive costs. The bridging target depends on the effort cost and the grid, because it is the highest level below the myopic threshold . $. 6 2 + \lambda / \gamma$ . With $\lambda / \gamma = . 1 0$ , a target of .60 no longer ends payments, and the organization must support practice up to .70, at a higher total cost. The total cost scales with $1 / \phi _ { u }$ and does not depend on $\phi _ { d }$ or H. The higher retention target costs less in every row with something to retain, except that both targets cost the same with $\phi _ { u } = . 0 5$ and the lower target costs less with $\lambda / \gamma = . 1 0$ . In both cases, the condition of Appendix A.9 does not hold strictly, because $\Delta u ( a _ { N } )$ is large relative to the gap one level down or because skill is regained quickly. The claim that a higher target can require less long-run spending holds in the running example and in most variations, but it is a possibility, not a general property, as Section 5.3 states.

Table 6: Running-example outcomes when one parameter is varied at a time. Novice boundary: largest $P _ { A I }$ at which the strategic worker solves at .50. Expert boundary: $a _ { N } - \lambda / \gamma$ . Trap width: their difference. Anna trapped: a strategic Anna at .50 accepts with the competent AI. Sam deskills: a strategic Sam at .80 accepts with the near-expert AI. Bridging target: cheapest target that ends payments for a novice starting at $a _ { 1 }$ with the competent AI, and its expected total cost. Retention: long-run payment per task to retain $a _ { N }$ or $a N _ { - 1 }$ with the near-expert $\mathrm { A I } ; - \mathrm { i f }$ Sam solves without an incentive. Strategic workers use $H = 1 0 0$ unless stated otherwise. On the grid with 9 levels, $\phi _ { u } = . 0 4$ and $\phi _ { d } = . 0 2$ The grids with diminishing and increasing steps keep the running-example values $\phi _ { u } = . 0 2$ and $\phi _ { d } = . 0 1$
<table><tr><td rowspan="2">Variation</td><td colspan="3">Boundaries in  $P _ { A I }$ </td><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">Bridging target</td><td colspan="2">Retention</td></tr><tr><td>Novice</td><td>Expert</td><td>Trap width</td><td>Sam deskills</td><td>Target</td><td>Cost</td><td> $a _ { N - 1 }$ </td></tr><tr><td>Running example</td><td>.590</td><td>.75</td><td>.160</td><td>yes</td><td>yes</td><td>.60</td><td>25.50</td><td>.020</td><td>.040</td></tr><tr><td> $\lambda / \gamma = . 0 2 5$ </td><td>.615</td><td>.775</td><td>.160</td><td>yes</td><td>no</td><td>.60</td><td>21.75</td><td></td><td></td></tr><tr><td> $\lambda / \gamma = . 1 0$ </td><td>.540</td><td>.70</td><td>.160</td><td>yes</td><td>yes</td><td>.70</td><td>34.00</td><td>.070</td><td>.057</td></tr><tr><td> $\phi _ { u } = . 0 1$ </td><td>.537</td><td>.75</td><td>.213</td><td>yes</td><td>yes</td><td>.60</td><td>51.00</td><td>.020</td><td>.060</td></tr><tr><td> $\phi _ { u } = . 0 5$ </td><td>.660</td><td>.75</td><td>.090</td><td>no</td><td>yes</td><td>.60</td><td>10.20</td><td>.020</td><td>.020</td></tr><tr><td> $\phi _ { d } = . 0 0 5$ </td><td>.590</td><td>.75</td><td>.160</td><td>yes</td><td>yes</td><td>.60</td><td>25.50</td><td>.020</td><td>.024</td></tr><tr><td> $\phi _ { d } = . 0 2$ </td><td>.590</td><td>.75</td><td>.160</td><td>yes</td><td>yes</td><td>.60</td><td>25.50</td><td>.020</td><td>.060</td></tr><tr><td> $H = 5 0$ </td><td>.536</td><td>.75</td><td>.214</td><td>yes</td><td>yes</td><td>.60</td><td>25.50</td><td>.020</td><td>.040</td></tr><tr><td> $H = 2 0 0$ </td><td>.645</td><td>.75</td><td>.105</td><td>no</td><td>yes</td><td>.60</td><td>25.50</td><td>.020</td><td>.040</td></tr><tr><td> $H = 1 0 0 0$ </td><td>.722</td><td>.75</td><td>.028</td><td>no</td><td>yes</td><td>.60</td><td>25.50</td><td>.020</td><td>.040</td></tr><tr><td>9 levels, step .05</td><td>.597</td><td>.75</td><td>.153</td><td>yes</td><td>yes</td><td>.65</td><td>21.75</td><td>.020</td><td>.023</td></tr><tr><td>diminishing steps</td><td>.577</td><td>.75</td><td>.173</td><td>yes</td><td>yes</td><td>.59</td><td>26.00</td><td>.020</td><td>.027</td></tr><tr><td>increasing steps</td><td>.582</td><td>.75</td><td>.168</td><td>yes</td><td>yes</td><td>.58</td><td>38.00</td><td>.020</td><td>.047</td></tr></table>

What the analysis does not vary. The analysis varies parameters within the model, not its structure. Skill still moves at most one level per task, learning requires independent solving, the AI’s accuracy is fixed, and workers know their own accuracy. Unequally spaced grids capture learning whose returns change with skill. Learning and decay probabilities that differ between levels leave the strategic decisions unchanged in form (Appendix A.7), but the incentive costs then depend on the probability at each supported level. Section 6.5 discusses these assumptions.