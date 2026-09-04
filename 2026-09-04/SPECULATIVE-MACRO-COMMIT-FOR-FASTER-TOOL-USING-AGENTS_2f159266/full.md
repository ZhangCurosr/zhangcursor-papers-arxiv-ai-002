# SPECULATIVE MACRO COMMIT FOR FASTER TOOL-USING AGENTS

Zeyu Liu<sup>⋆</sup> Souvik Kundu<sup>†</sup> Peter A. Beerel<sup>⋆</sup>

<sup>⋆</sup> University of Southern California, Los Angeles, USA <sup>†</sup> Intel Labs, San Diego, USA

## ABSTRACT

Tool-using LLM agents spend wall-clock time not only on model inference but also in serial action–observation turns, where each tool call, environment transition, and observation can delay subsequent decisions. We introduce Speculative Macro Commit (SMC), a runtime mechanism for a two-tier agent system: a large authoritative actor model produces the official trajectory, while a faster speculative drafter model continuously predicts and executes future action chains on an isolated environment snapshot. SMC mines recurring multiaction skeletons from training traces and stores them in a macro library used to match against action chains predicted by the drafter at runtime. When the actor’s next tool call matches the first drafted action, SMC commits the remaining pre-executed draft steps, together with their observations, to the official trajectory. Using Qwen3.5-27B INT4 as the authoritative actor model and Qwen3.5-4B as the speculative drafter model, SMC matches the sequential agent’s overall accuracy while reducing latency by 10.23% over the Speculative Actions (SA) baseline and 18.59% over sequential execution on the τ<sup>2</sup>-Bench Telecom subset. On AppWorld, SMC reduces wall time by 7.7% over SA baseline and 44.9% over sequential execution, with a small reduction in task completion. Overall, SMC provides a practical way to reuse multi-step speculative execution and reduce agent latency beyond single-step speculative actions. Our code is publicly available here.

Index Terms— LLM Agents, speculative execution, inference latency

## 1. INTRODUCTION

Language agents are becoming a practical interface to software, data, web, and customer-service workflows [1–3]. Unlike a model that makes a single prediction, a tool agent runs a loop: it calls the model to choose an action, executes that action in an environment, observes the result, and conditions its next decision on what it observed [4]. This iterative interaction gives agents much of their flexibility, but it also creates a wallclock bottleneck that model throughput alone does not capture: a single task can wait through dozens of sequential turns even when the model server is well batched.

This latency has several sources. Repeated large-model calls reprocess long contexts and stress cache management [5]. Tool orchestration adds scheduling, state, and environmentlifecycle overhead [6]. The tools and environments can themselves dominate a request; PASTE reports tool execution at 35% to 61% of total latency in representative agents [7]. Across these sources, the main constraint is sequential dependence: the next prompt usually cannot be built until the previous action returns its observation. Token-level speculative decoding [8] drafts several tokens with a cheap model and verifies them with the expensive one, but it operates strictly inside one model call and cannot remove this cross-step dependency.

Speculative Actions (SA) lifts the draft-then-verify idea from tokens to actions [9]. A fast speculative drafter predicts likely future actions while a slower authoritative actor computes the next verified action; when the actor later agrees, an action that was already launched can be committed. However, the core SA mechanism only accepts progress at the granularity of adjacent action steps: a correct guess lets the executor reuse work for the next step, while naively longer lookahead has limited benefits. Therefore, how to effectively commit multi-step actions still remains underexplored. A complementary line of work observes that tool-agent traces contain repeated sequences of tool calls. AWO [10], for example, mines recurring tool-call sequences and turns them into deterministic composite meta-tools that bundle multiple actions into one invocation. This can remove several reasoning steps when the model chooses the right meta-tool. In our experiments, however, simply exposing mined action sequences as additional callable tools is unreliable: open models like Qwen3.5-27B [11] rarely select these mined macros. This raises the central question of the paper:

## Can recurring action patterns reduce agent la tency without requiring the model to choose new meta tools?

Our answer is Speculative Macro Commit (SMC), where a macro is a recurring multi-action pattern mined from execution traces. A commit means accepting already executed draft tool calls and their returned results into the agent’s real execution history, so that the large model does not need to choose and execute those steps again. When one macro from the mined macro library is identified in this long action chain, and the first action of the macro matches the authoritative action, SMC attempts a macro commit: it commits the remaining matched draft steps and their observations into the official trajectory, skipping the corresponding large-model calls as well as environment delays. As a macro commit does not ask the actor to regenerate and verify every skipped step, SMC is approximate rather than strictly lossless but is empirically shown to preserve task quality while reducing latency.

1. An end-to-end runtime for action-level speculation. We build a two-tier runtime in which an authoritative actor model and a speculative drafter model run concurrently throughout complete long-horizon tool-agent tasks. The executor measures actual wall-clock latency, lets the drafter run ahead on draft action–observation chains, and falls back to ordinary execution when speculation cannot be used.

2. Speculative Macro Commit. We introduce SMC, which shifts mined recurring action patterns from model-side macro selection to executor-side validation. The executor first identifies macro matches in the drafter’s pre-executed action chain; after the authoritative actor agrees with the first matched draft action, SMC can commit the remaining matched draft steps and their observations into the official trajectory. This skips the corresponding large-model calls and avoids waiting again for already returned tool results, making SMC an approximate but guarded latency optimization.

3. Evaluation on full tool-agent benchmarks. With Qwen3.5-27B INT4 as the authoritative actor and Qwen3.5-4B as the speculative drafter on the τ<sup>2</sup>-bench Telecom subset, the SMC run matches the single-model baseline’s reward while cutting latency by 18.59%; and reduces latency by 10.23% over SA. On AppWorld benchmark, SA reduces the latency by 40.37% compared with the baseline, and SMC further reduces wall time by 7.64% with a small accuracy tradeoff.

## 2. RELATED WORK

Speculative decoding and speculative agent execution. Speculative decoding accelerates a single model call by letting a cheap drafter model propose tokens that a target model verifies in batches [8]. This line of work reduces decoding latency inside one model invocation, but a ReAct-style tool agent, which interleaves model reasoning with tool calls over many steps [4], also pays latency across invocations, since each tool result must return before the next prompt can be formed. Speculative Actions extends the draft-and-verify idea from tokens to agent actions, launching predicted future API calls in parallel with an authoritative actor and committing them only when predictions match [9]. Our SMC shares this action-level view but differs in what a commit covers: rather than verifying one action at a time, it commits several already executed draft steps, trading the lossless guarantee of one-step verification for a guarded multi-step approximation.

SMC builds on the same actor–speculator setting but uses a different commit rule. After the actor verifies the first drafted action, SMC may commit a matched suffix of already executed action–observation pairs without verifying each skipped action with the actor. This enables multi-step runtime skipping, at the cost of making SMC approximate rather than strictly lossless.

Speculative tool execution systems. Recent LLM-serving systems attack the same serial LLM–tool loop from complementary angles. PASTE speculatively executes likely tool calls while the model is still computing, exploiting stable application-level control flow and data dependencies [7], while ThunderAgent [6] and KVFlow [5] reduce latency through program-aware serving and prefix/cache management without changing which actions enter the trajectory. These systems motivate the runtime setting of our work: latency depends on whether useful tool work can move off the serial critical path, not only on model calls.

Workflow compression and meta-tools. Agent Workflow Optimization (AWO) mines recurrent tool-call sequences from traces and registers the resulting composites as meta-tools available to the model [10]. The key difference is the interface: AWO changes the model-visible toolset and lets the model choose when to call a composite, whereas our SMC keeps the mined steps as hidden runtime state and commits them only after the anchor call is verified.

## 3. SPECULATIVE MACRO COMMIT

## 3.1. Executor Roles and State

SMC is implemented in the agent executor, the program that runs the tool-agent loop. Given the current task history, the executor prompts the model for the next tool call, executes that call, records the returned result, and repeats this process until the task finishes or the step budget is exhausted. The executor is therefore distinct from both the language model and the tool environment: it alone decides which action–result pairs become part of the task history seen by future model calls.

SMC augments this executor with two models that run concurrently. The authoritative actor is the large model whose tool calls define the real execution of the agent. The speculative drafter is a smaller and faster model that runs ahead of the actor, trying to speculate short future tool-call sequences from the committed history H. The executor maintains H, consisting of the tool calls and returned results that future actor prompts will condition on, and a live task state E, which is the state reached by executing the calls in H. The drafter’s calls are executed in an isolated draft state E<sup>i</sup>, so they produce returned results without immediately changing the live task state. We write the drafter’s current output as a draft chain

$Q = ( q _ { 1 } , \dotsc , q _ { D } )$ , where each $q _ { i }$ contains a drafted tool call and the result of executing it in the draft state. A macro is a recurring multi-action pattern mined from past executions. A commit is the executor operation that appends executed tool calls and results pairs to H and advances E consistently with them. When no macro commit is performed, H grows in the usual one-step way: the executor appends the authoritative actor’s tool call and its returned result. SMC changes this loop through a single addition, the macro commit rule of Section 3.3, which under specific conditions lets one actor step be followed by several already executed draft steps.

When the macro commit rule is disabled, the same executor still runs the actor and drafter concurrently and still reuses single-step speculative work: whenever the actor proposes the same call as the first drafted step, the drafter’s already executed call and result are committed directly. We call this configuration the SA-only baseline and use it as the matchedruntime comparison in Section 4.

## 3.2. Macro Mining

SMC first extracts candidate multi-action patterns from successful training traces. It then evaluates these candidates using drafter predictions from sampled training states and keeps only those that satisfy the selection criteria below. The selected candidates form the runtime macro library M. Each macro records an ordered tool-call sequence together with the argument structure needed to match it, replacing task-specific values such as user IDs, document IDs, or timestamps with slots. This normalization lets one macro cover repeated behavior across tasks without requiring identical concrete arguments.

Because a macro is useful only if the drafter can reproduce it at deployment, we label candidates under the same actor– drafter setting used at runtime. For each sampled training state, the drafter proposes a short future action chain, which we compare against a reference future trajectory from that state. A proposal is labeled correct when its tool-call sequence matches the reference future trajectory after task-specific argument values are normalized as described above. Each candidate macro thus carries two kinds of evidence: how often the pattern occurs, and how reliably the drafter reproduces it when the relevant state is encountered.

For a candidate macro m, let $n _ { m }$ be the number of labeled opportunities and $k _ { m }$ be the number of correct drafter proposals. We compute a conservative reliability estimate using the lower δ-quantile of a Beta posterior,

$$
\underline { { p } } _ { m } = F _ { \mathrm { B e t a } ( k _ { m } + 1 , ~ n _ { m } - k _ { m } + 1 ) } ^ { - 1 } ( \delta )\tag{1}
$$

and retain m only when $n _ { m } ~ \ge ~ n _ { \mathrm { m i n } }$ and $\underline { { { p } } } _ { m } \ \geq \ \tau$ . The thresholds $n _ { \mathrm { m i n } }$ and τ are selected on held-out tasks using the same executor configuration as the final evaluation.

These criteria are used only to filter the candidate macros. Whether a retained macro is committed is determined separately at runtime by the macro commit rule.

## 3.3. Macro Commit Rule

At runtime, the drafter continuously extends the draft chain Q from the current H. Consider a retained macro $m =$ $( u _ { 1 } , \dotsc , u _ { K } )$ . A runtime alignment matches $u _ { 1 } , \ldots , u _ { j } \ { \mathrm { t o } }$ a suffix of H and the remaining $\ell = K - j$ actions to the draft prefix $q _ { 1 } , \ldots , q _ { \ell }$ . The library stores m once; j and ℓ are determined by the state reached at runtime. An alignment is only a commit candidate; it does not change H by itself. The executor waits for the actor’s next tool call. If that call matches q , we call $q _ { 1 }$ the anchor call. The drafter has already executed $q _ { 1 }$ so the executor commits it together with the drafter’s returned result; only $q _ { 2 } , \ldots , q _ { \ell }$ are committed without individual actor confirmation. This skips ℓ − 1 actor decisions, so candidate selection requires $\ell - 1 \geq L _ { \mathrm { m i n } }$ . The floor cannot be enforced from K alone: for the same four-action macro, aligning one versus two actions with H leaves three versus two draft actions and therefore skips two versus one actor decisions.

Online checks then discard stale draft work and apply benchmark-specific action safety. In AppWorld, manually specified API-name rules reject irreversible or unknown calls, while forkable mutations require a matching live-state replay. If all checks pass, the executor appends $q _ { 2 } , \ldots , q _ { \ell }$ and their returned results to H and advances E accordingly. The drafter is restarted only when the committed history actually diverges from the draft chain, that is, when the actor’s call does not match $q _ { 1 }$ . In that case the executor executes the actor’s call in the live state, waits for its result, commits this step, discards the draft work, and restarts the drafter from the updated history. Whenever the actor confirms $q _ { 1 }$ , the executor instead reuses the drafter’s execution: it removes the committed steps from $Q \left( q _ { 1 } \right.$ alone if no macro commit, $q _ { 1 } , \ldots , q _ { r }$ if one does), and the drafter keeps extending the remainder of the draft chain, which the executor continues to search for macro matches in later iterations. If candidate selection finds no eligible alignment (including the depth floor), or if an online check rejects the selected actions, only the anchor step is committed and the executor continues as the SA-only baseline. Algorithm 1 summarizes this rule.

Overall, unlike registered meta-tools, the actor model never emits a macro call: the actor sees the same tool schema as the baseline, and a macro step is a runtime decision over already executed drafter work, not a new action the model must learn.

## 4. EXPERIMENTAL RESULTS

## 4.1. Setup

Models and serving. The authoritative actor is a quantized Qwen3.5-27B model, sized to fit on a single GPU; the speculative drafter is Qwen3.5-4B [11], and both use greedy decoding. The sequential baseline runs only the actor on a single GPU. The two speculative runtimes use three GPUs: the actor, an actor-class replica that serves speculative requests so they do not queue behind the authoritative request stream, and the drafter. The SA-only runtime (hereafter SA) and SMC share this serving configuration and differ only in whether the macro commit rule of Section 3.3 is enabled.

Algorithm 1 SPECULATIVE MACRO COMMIT Executor   
Require: macro library M, live task state E, task x   
Require: authoritative actor $A _ { \mathrm { a c t o r } } .$ speculative drafter $A _ { \mathrm { d r a f t } }$   
Require: draft depth D, minimum skipped length $L _ { \mathrm { m i n } }$   
1: $H \gets [ ]$ ▷ committed task history   
2: launch $A _ { \mathrm { d r a f t } }$ to maintain a draft chain $Q = ( q _ { 1 } , \dotsc , q _ { D } )$ from   
$H$   
3: $F _ { \mathrm { a c t o r } } \gets \mathrm { S U B M I T A C T O R } ( A _ { \mathrm { a c t o r } } , H , E )$   
4: while task not finished and budget remains do   
5: ${ a } _ { \mathrm { a c t o r } } \gets \mathbf { W } \mathbf { A I T } ( F _ { \mathrm { a c t o r } } ) \triangleright$ actor’s proposed tool call (not yet   
executed)   
6: Q ← LATESTDRAFTCHAIN   
7: if FIRSTACTIONMATC $\operatorname { I } ( a _ { \mathrm { a c t o r } } , q _ { 1 } )$ then   
8: $\mathrm { C O M M I T } ( H , E , q _ { 1 } )$ ▷ reuse the drafter’s executed call   
and result   
9: $r \gets 1$   
10: $c \gets \mathrm { S E L E C T M A C R O } ( \mathcal { M } , H , Q , L _ { \operatorname* { m i n } } )$   
11: $\mathbf { i f } \ c \neq \perp$ then   
12: $( m , r ^ { \prime } ) \gets c$ ▷ m matches $q _ { 1 } , \ldots , q _ { r ^ { \prime } }$ and   
$r ^ { \prime } - 1 \geq L _ { \mathrm { m i n } }$   
13: if ONLINECHECKS(m, H, $( q _ { 2 } , \dots , q _ { r ^ { \prime } } ) , x ,$ E) then   
14: for $q \in \left( q _ { 2 } , \ldots , q _ { r ^ { \prime } } \right)$ do   
15: COMMIT $H , E , q )$   
16: end for   
17: $r \gets r ^ { \prime }$   
18: end if   
19: end if   
20: POPDRAFTPREFIX $( Q , r )$ ▷ drafter keeps extending the   
remainder of $Q$   
21: else   
22: o ← EXECUTE(a , E) ▷ run the actor’s call in the   
live state   
23: COMMIT(H, $E , ( a _ { \mathrm { a c t o r } } , o ) )$   
24: discard draft work; restart $A _ { \mathrm { d r a f t } }$ from H   
25: end if   
26: $F _ { \mathrm { a c t o r } } \gets \mathrm { S U B M I T A C T O R } ( A _ { \mathrm { a c t o r } } , H , E )$   
27: end while

Benchmarks and metrics. We evaluate on two full toolagent benchmarks, $\tau ^ { 2 }$ Telecom [12] and AppWorld [13]. We use binary task accuracy for $\tau ^ { 2 }$ Telecom and task-goal completion (TGC) for AppWorld as performance metrics, and average per-task wall time as the latency metric.

## 4.2. Main Results

As shown in Table 1 , on $\tau ^ { 2 }$ Telecom, SA reduces average latency from 27.60s to 25.03s per task, 9.31% below the sequential baseline, and SMC further reduces it to 22.47s, 18.59% below the baseline and 10.23% below SA. The paired outcomes are unchanged: SMC agrees with the sequential baseline on every task (2,274 correct and 11 incorrect under both runs), and relative to SA it fixes one task and regresses none. On this benchmark, macro commit converts committed draft steps into latency savings without changing any task outcome.

Table 1. Full benchmark results. Latency is average seconds per task; $\Delta$ is relative to the sequential baseline within each benchmark. $\tau ^ { 2 } .$ -Bench accuracy is binary task accuracy; App-World accuracy is TGC.
<table><tr><td>Benchmark</td><td>Run</td><td>Acc.</td><td>Lat.</td><td>∆(%)</td></tr><tr><td> $\tau ^ { 2 }$  Telecom</td><td>Baseline</td><td>99.52</td><td>27.60</td><td></td></tr><tr><td></td><td>SA</td><td>99.47</td><td>25.03</td><td>-9.31</td></tr><tr><td></td><td>SMC</td><td>99.52</td><td>22.47</td><td>-18.59</td></tr><tr><td>AppWorld</td><td>Baseline</td><td>41.67</td><td>355.7</td><td></td></tr><tr><td></td><td>SA</td><td>41.67</td><td>212.1</td><td>-40.37</td></tr><tr><td></td><td>SMC</td><td>40.48</td><td>195.9</td><td>-44.93</td></tr></table>

Table 2. Macro-step coverage in the main benchmark runs. Skip density divides skipped steps by tasks × max\_steps.
<table><tr><td>Benchmark</td><td>Commit rate</td><td>Hits</td><td>Skipped</td><td>Skip density</td></tr><tr><td>AppWorld</td><td>62.0%</td><td>219</td><td>512</td><td>3.81%</td></tr><tr><td> $\tau ^ { 2 }$  Telecom</td><td>86.2%</td><td>3,352</td><td>7,154</td><td>3.91%</td></tr></table>

AppWorld shows the same direction in a harder regime. SA preserves TGC while reducing latency by 40.37% relative to the sequential baseline, and SMC reaches to 195.9s per task, 44.93% below the baseline and 7.64% below SA. Unlike $\tau ^ { 2 } .$ Bench, this additional speedup comes with a small TGC drop, from 70/168 to 68/168 tasks.

## 4.3. Analysis of the AppWorld Gain

The smaller AppWorld gain over SA is not explained by a lack of macro opportunities. Table 2 reports skip density, defined as the number of skipped steps divided by total number of steps in the benchmark. On AppWorld, SMC commits at least one macro on 62.0% of tasks and successfully skips 512 actor steps, yielding a skip density of 3.81%, close to the 3.91% observed on $\tau ^ { 2 }$ Telecom. This similarity indicates that, after normalizing for differences in dataset size and step budget, SMC commits a comparable amount of skipped work on both benchmarks. The two benchmarks therefore differ not in whether SMC finds reusable patterns, but in how reliably the saved work survives into end-to-end wall time.

AppWorld is the harder case because task completion, retries, and the per-task step cap (Section 4.1) all interact with latency. An all-task average therefore mixes three effects: skipped draft steps on trajectories that proceed as before, trajectories whose outcome changes, and steps that consume budget without issuing a tool call. Table 3 controls for the latter two. On the 143 tasks where SA and SMC reach the same correctness outcome, SMC reduces total wall time by 13.5%, substantially more than the 7.64% all-task gain. On the 85 tasks where neither run emits a no-tool-call (NTC) step, the gain is 10.7%. This slice removes idle, recovery-heavy trajectories whose wasted steps inflate wall time and dilute the macro saving It does not hide a side effect of SMC, as SMC produces fewer such steps than SA (68 NTC steps over 31 tasks, versus 105 over 78 for SA), so macro commit reduces rather than induces idle recovery. On comparable trajectories, macro commit thus converts skipped draft steps into latency reduction more effectively than the aggregate number suggests; the all-task figure is diluted by outcome shifts and by the idle trajectories isolated above.

Table 3. AppWorld controlled wall-time slices.
<table><tr><td>Subset</td><td>SA</td><td>SMC</td><td>∆(%)</td></tr><tr><td>Same-acc.</td><td>28,920s</td><td>25,014s</td><td>-13.5</td></tr><tr><td>NTC-free</td><td>16,375s</td><td>14,625s</td><td>-10.7</td></tr></table>

## 5. MECHANISM ABLATIONS

## 5.1. Ablations

The main results suggest that a macro commit helps only when it satisfies the three conditions built into the commit rule of Section 3.3. The reused steps must stay hidden from the model, since asking the model to select mined patterns adds a decision problem rather than skipping any actor call; the runtime must filter mined candidates aggressively, since a library match alone weakly predicts that the trajectory will follow the mined pattern; and a commit must skip enough actor calls on the critical path to overcome speculation, verification, and synchronization overhead. The three ablations below test these conditions in turn.

## 5.2. Interface: hidden runtime state

Table 4 evaluates two simpler ways of using mined routines on a single GPU. The first exposes mined patterns as registered meta-tools [10], making reuse a model-visible action. The second leaves the model interface unchanged but commits matched patterns passively, without the online checks of the full commit rule.

Neither alternative has desirable results. Registered metatools slightly increase latency and lower accuracy, because the actor almost never selects them: exposing mined routines as ordinary tools does not reliably skip actor calls, it poses an extra tool-selection problem. Passive committing has the opposite failure mode. It cuts latency by 11.34%, confirming that hidden runtime reuse can skip actor calls, but accuracy falls from 99.52% to 96.48%. Hidden reuse alone is therefore insufficient: the runtime must also verify that the committed steps are anchored to the authoritative trajectory. SMC keeps the latency benefit of hidden reuse while adding the online commit conditions.

Table 4. Results on $\tau ^ { 2 } .$ -Bench with single GPU setting.
<table><tr><td>Run</td><td>Acc.</td><td>Lat.</td><td>∆(%)</td></tr><tr><td>Baseline</td><td>99.52</td><td>27.60</td><td>一</td></tr><tr><td>Baseline + AWO-like</td><td>99.34</td><td>27.89</td><td> $+ 1 . 0 5$ </td></tr><tr><td>Baseline + passive</td><td>96.48</td><td>24.47</td><td> $- 1 1 . 3 4$ </td></tr></table>

Table 5. Conditional precision of the commit stages on 100 held-out $\tau ^ { 2 }$ Telecom tasks. Rows above the rule are event-level exact-match checks; the committed row is a task-outcome check.
<table><tr><td>Filter stage</td><td>Events</td><td>Correct / outcome-preserving</td></tr><tr><td>Library match only</td><td>1,968</td><td>681/1,968 (34.6%)</td></tr><tr><td>+ drafter-executed</td><td>885</td><td>625/885 (70.6%)</td></tr><tr><td>+ verified anchor call</td><td>711</td><td>625/711 (87.9%)</td></tr><tr><td>+ depth guard  $( L _ { \operatorname* { m i n } } = 1 )$ </td><td>343</td><td>310/343 (90.4%)</td></tr><tr><td>Actually fired</td><td>158</td><td>158/158 (100.0%)</td></tr></table>

## 5.3. Commit Precision

The passive-committing result raises the central safety question: how does SMC avoid committing incorrect mined patterns? Table 5 answers with a staged audit on 100 held-out $\tau ^ { 2 }$ Telecom tasks. For the first four rows, SMC suppresses the commit and lets the actor continue, so the realized trajectory provides an event-level counterfactual for whether the candidate steps were exact; the final row reports task-outcome preservation for the steps SMC actually committed.

The mined library by itself is far too noisy to act as a commit rule: a library match reproduces the exact future steps in only 34.6% of events. Requiring the drafter to have already executed the steps raises precision to 70.6%, since SMC no longer commits patterns that exist only as offline matches. Verifying the anchor call raises it to 87.9% by tying the candidate to the authoritative trajectory, and the depth guard $( L _ { \operatorname* { m i n } } = 1$ at least one committed step after the anchor) raises it to 90.4% while removing shallow commits unlikely to repay synchronization overhead. Under full rule, all committed events preserve the outcome in this audit. As noted in Section 3.3, this is not a proof that every committed pattern is uniquely correct; it shows that the online checks turn a low-precision library into a high-precision set of commits on this held-out audit.

## 5.4. Critical-Path Depth

High precision alone does not guarantee speedup: a correct macro is unhelpful if it skips only one or two actor calls or skips calls that are off the critical path. Table 6 compares the final SMC runtime with an earlier runtime that committed more aggressively. The legacy runtime commits almost twice as often as the final runtime and skips more total steps, yet it is 1.64% slower than SA. Raw macro count is thus the wrong objective: many additional legacy commits are too shallow or poorly aligned with already executed draft work, adding commit overhead without reducing end-to-end latency. The final runtime suppresses 6,281 depth-one opportunities, enforces the $L _ { \mathrm { m i n } }$ depth floor, and commits only steps already executed on the draft branch. While it commits fewer macros and skips fewer total steps, it is 10.23% faster than SA. The useful unit is therefore not a macro hit but a guarded, sufficiently deep commit that skips actor calls on the critical path.

Table ${ \bf 6 . \nabla } \tau ^ { 2 }$ Telecom full-run comparison: raw macro count versus critical-path commits. Hits is the number of macro commits; Skip is the total committed draft steps after anchors.
<table><tr><td>Run</td><td>Acc.</td><td>Lat.</td><td>Hits</td><td>Skip</td><td>∆(%)</td></tr><tr><td>SA</td><td>99.47</td><td>25.03</td><td>0</td><td>0</td><td>一</td></tr><tr><td>Legacy macro</td><td>99.56</td><td>25.44</td><td>6,410</td><td>10,528</td><td>+1.64</td></tr><tr><td>Final SMC</td><td>99.52</td><td>22.47</td><td>3,352</td><td>7,154</td><td>-10.23</td></tr></table>

## 6. CONCLUSION

We presented Speculative Macro Commit (SMC), a runtime mechanism that extends speculative action execution from single-step reuse to committing several already executed draft steps after a verified anchor call. Because a macro commit is approximate rather than losslessly verified, SMC guards it with conservative offline mining, a verified anchor call, a minimum skipped depth, and online state and argument checks. $\mathrm { O n } \tau ^ { 2 }$ Telecom, SMC gives a quality-preserving latency gain over the full dataset, beyond both the sequential baseline and an equal-hardware speculation-only baseline (SA); on AppWorld it gives a larger wall-time gain with a small completion tradeoff, and larger gains on same-accuracy slices. More broadly, workflow macros can live as hidden runtime state rather than model-visible tools, but they help only when the pattern is predictable enough to mine, the drafter has already executed it in the current run, and the commit survives the online checks.

## References

[1] Haoyuan Xu, Chang Li, Xinyan Ma, Xianhao Ou, Zihan Zhang, Tao He, Xiangyu Liu, Zixiang Wang, Jiafeng Liang, Zheng Chu, Runxuan Liu, Rongchuan Mu, Dandan Tu, Ming Liu, and Bing Qin, “The evolution of tool use in llm agents: From single-tool call to multi-tool orchestration,” 2026.

[2] George Ling, Shanshan Zhong, and Richard Huang, “Agent skills: A data-driven analysis of claude skills for extending large language model functionality,” 2026.

[3] Tianxin Wei, Ting-Wei Li, Zhining Liu, Xuying Ning,

Ze Yang, Jiaru Zou, Zhichen Zeng, Ruizhong Qiu, Xiao Lin, Dongqi Fu, Zihao Li, Mengting Ai, Duo Zhou, Wenxuan Bao, Yunzhe Li, Gaotang Li, Cheng Qian, Yu Wang, Xiangru Tang, Yin Xiao, Liri Fang, Hui Liu, Xianfeng Tang, Yuji Zhang, Chi Wang, Jiaxuan You, Heng Ji, Hanghang Tong, and Jingrui He, “Agentic reasoning for large language models,” 2026.

[4] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao, “ReAct: Synergizing reasoning and acting in language models,” in International Conference on Learning Representations (ICLR), 2023.

[5] Zaifeng Pan, AJJKUMAR PATEL, Yipeng Shen, Zhengding Hu, Yue Guan, Wan-Lu Li, Lianhui Qin, Yida Wang, and Yufei Ding, “KVFlow: Efficient prefix caching for accelerating LLM-based multi-agent workflows,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

[6] Hao Kang, Ziyang Li, Xinyu Yang, Weili Xu, Yinfang Chen, Junxiong Wang, Beidi Chen, Tushar Krishna, Chenfeng Xu, and Simran Arora, “Thunderagent: A simple, fast and program-aware agentic inference system,” 2026.

[7] Yifan Sui, Han Zhao, Rui Ma, Zhiyuan He, Hao Wang, Jianxun Li, and Yuqing Yang, “Act while thinking: Accelerating llm agents via pattern-aware speculative tool execution,” 2026.

[8] Yaniv Leviathan, Matan Kalman, and Yossi Matias, “Fast inference from transformers via speculative decoding,” in International Conference on Machine Learning. PMLR, 2023, pp. 19274–19286.

[9] Naimeng Ye, Arnav Ahuja, Georgios Liargkovas, Yunan Lu, Kostis Kaffes, and Tianyi Peng, “Speculative actions: A lossless framework for faster AI agents,” in The Fourteenth International Conference on Learning Representations, 2026.

[10] Sami Abuzakuk, Anne-Marie Kermarrec, Rishi Sharma, Rasmus Moorits Veski, and Martijn de Vos, “Optimizing agentic workflows using meta-tools,” 2026.

[11] Qwen Team, “Qwen3.5: Towards native multimodal agents,” February 2026.

[12] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan, “τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment,” 2025.

[13] Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian, “AppWorld: A controllable world of apps and people for benchmarking interactive coding agents,” in ACL, 2024.