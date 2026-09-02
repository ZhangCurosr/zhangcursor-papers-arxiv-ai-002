# REVISE: Validity-Guided Recovery for Online Revisions in Agent Workflows

Ruoling Qi<sup>1,2</sup> Xuaner Wu<sup>1</sup> Penghang Liu Jian Chen Yirui Liu<sup>1,†</sup>

<sup>1</sup>Institute of Artificial Intelligence, China Telecom (TeleAI)

<sup>2</sup>Shanghai Jiao Tong University

<sup>†</sup>Corresponding author

Ruoling Qi: qiruoling760@sjtu.edu.cn Yirui Liu: yiruiliu926@gmail.com

## Abstract

Agent revisions expose a fundamental correctness–efficiency trade-off during concurrent execution. Discarding ongoing work preserves latest-version correctness but wastes progress that may remain valid, whereas reusing prior work preserves efficiency but risks propagating stale state into outputs and tool effects. Existing recovery strategies resolve this trade-off in an imbalanced way with coarse-grained policies: they either favor efficiency by allowing potentially stale work to continue, or favor correctness by restarting the workflow or recomputing a linear suffix from the earliest conflict, thereby discarding unaffected progress. We present REVISE, a validity-guided runtime for fine-grained recovery in structured agent workflows. When a revision arrives, REVISE first intersects its delta with recorded data and control dependencies and propagates the resulting impact through the partially executed DAG to identify affected work. It then stops invalid work, preserves validity-established progress beyond the earliest conflict, and recomputes only the affected region. Incomplete provenance conservatively expands recovery, while reused results are revalidated before commit. Analysis of real coding-agent traces show online recovery opportunities: 118 sessions retain observable work before a queued later message is delivered; across 167 overlapping assistant responses, enqueue-to-completion overlap reaches 56.55 s at p95. Across 300 challenging revision/commit executions, REVISE matches a latest-version oracle with no stale outputs or effects. On unmodified LangGraph and LLMCompiler applications using Qwen3-14B, it reduces model calls by 40.6–56.0% relative to full restart and by 31.3–43.6% relative to suffix recomputation. Under serving pressure, it further reduces revision-to-correct-completion tokens by 13.26% and improves SLO goodput by 3.07–5.43%.

## 1 Introduction

Multi-turn agent workflows are becoming increasingly interactive: users can provide new instructions, corrections, or clarifications while model and tool calls from an earlier turn are still in flight [1]. These inputs may change requirements, intermediate results, or control decisions on which ongoing or completed tasks depend. We call such an event a revision. When a revision arrives during ongoing agent execution, the runtime faces a fundamental correctness–efficiency trade-off. Delaying intervention allows potentially stale in-flight work to continue, wasting computation after the revision arrives; full restart preserves latest-version correctness but discards all prior progress; and recomputing a linear suffix from the earliest conflict is more selective yet can still redo unaffected work beyond that point. The central challenge is therefore to determine fine-grained execution validity after a revision, so invalid work can be stopped while valid progress is preserved, balancing latest-version correctness with recovery efficiency.

Existing systems address different aspects of this trade-off, but do not provide a mechanism for finegrained recovery after online revisions. Workflow and serving runtimes schedule structured model and tool execution [2, 3, 4], but do not determine how a revision changes the validity of partially executed work. Revisable execution identifies an earliest conflict and rolls back the subsequent trace [5], but its linear recovery boundary can discard unaffected work beyond that conflict. Transactional runtimes protect externally visible effects [6, 7], but do not decide which prior computation can be safely retained. What remains unresolved is a fine-grained recovery mechanism over a partially executed DAG: which work is invalidated, and which progress can still be preserved.

We present REVISE, a validity-guided runtime for fine-grained recovery in structured agent workflows. When a revision arrives, REVISE intersects its delta with recorded data and control dependencies and propagates the resulting impact through the executed DAG to identify affected work. It then stops invalid work, preserves proven-valid progress beyond the earliest conflict, and recomputes only the affected region. Fine-grained provenance enables selective recovery; when provenance is incomplete, REVISE conservatively expands recovery toward a suffix or full restart. Reuse remains provisional: outputs and staged tool effects are revalidated against intervening revisions before commit.

Analysis of real coding-agent traces shows opportunities for online intervention: among 5,825 analyzable sessions, 118 retain observable work before a queued later message is delivered; across 167 overlapping assistant responses, enqueue-to-completion overlap reaches 56.55 s at p95. Across 300 adversarial executions, REVISE matches a latest-version oracle without stale outputs or effects. On unmodified LangGraph and LLMCompiler applications using Qwen3-14B, it reduces model calls by 40.6–56.0% relative to full restart and by 31.3–43.6% relative to suffix recomputation. Under GPU serving pressure, it further reduces revision-to-completion tokens by 13.26% and improves SLO goodput by 3.07–5.43%. These results show that timely intervention avoids obsolete work, while provenance-guided selective recovery preserves additional progress when revisions are local.

In summary, our contributions are as follows:

• an online execution validity abstraction for determining which completed, running, and pending work remains valid after a revision;

• a validity-guided, fail-closed recovery protocol that identifies affected work from revision deltas and dynamic data/control dependencies, selectively preserves valid progress, and revalidates outputs and effects before commit;

• an evaluation demonstrating real online revision opportunities, latest-version correctness across challenging event orderings, and lower recomputation with serving gains under contention.

## 2 Revision opportunities and the limits of offline evidence

A revision can create two distinct recovery opportunities in a partially executed workflow. Temporal overlap allows obsolete in-flight work to be stopped early, while locality allows valid progress outside the affected region to be preserved. We therefore ask two empirical questions: Do revisions arrive early enough to intervene? and Do real revisions exhibit locality that could permit partial recovery? Figure 1 illustrates the desired behavior: stop obsolete work, preserve progress whose validity can be established, and recompute only the affected paths.

To answer the first question, we need both conversational semantics and an execution timeline. SWEchat [1] provides real coding-agent sessions from which we reconstruct when later user messages were queued relative to ongoing assistant, tool, and background work. Static issue-resolution benchmarks such as SWE-bench [8] cannot establish such intervention windows because they contain neither interleaved user revisions nor an execution timeline. Timing alone, however, does not imply invalidation: a later message may be an ordinary follow-up rather than a revision. We therefore combine timestamp reconstruction with manual semantic audit.

The traces reveal a real but long-tailed intervention opportunity. Among 5,825 analyzable SWE-chat sessions, 657 contain a matched queued-revision signal; 118 of these retain observable assistant, tool, or background work before the revision is delivered. These sessions contribute 174 work-bearing queued-revision events, and 62 sessions expose an intervention window of at least 10 s. Assistant responses completed within these windows overlap for 5.54 s at the median, 56.55 s at p95, and 853.16 s at the maximum. These measurements establish temporal opportunity: revisions can arrive early enough for a runtime to stop ongoing work. They do not establish how much computation is avoidable, because the traces lack token-level progress and dependency-complete execution state.

![](images/4106b4de532deedd14d4467a8821ea57e1e57d11ed0df4ae00aac673f0433911.jpg)

![](images/225bb486eef68502ee1fc231a2e4222c07157e33a99a5887fb840fce107971a0.jpg)  
Figure 1: Desired behavior at revision time $t _ { r }$ . A revision-unaware runtime cannot safely distinguish obsolete from still-valid work. REVISE aims to stop affected work, preserve valid progress, and recompute only the affected region.

![](images/c7181d528f1b97d83ff6f2f01ce7d070ae35fffb5f5f219b8929ceb0bddea05c.jpg)

![](images/218604c39041e90dec84f0c8f7b620324396d23170fe5f6cef001a147c47a6ac.jpg)

![](images/3339e789b9de33fe0a899884a66894993d333531f5212281c754b2ea53e7edf9.jpg)  
Figure 2: SWE-chat evidence for revision opportunity: (a) the trace funnel, (b) overlap duration, and (c) locality among manually audited revisions.

We next examine whether these revisions appear local enough to permit partial recovery. From the 174 work-bearing events, we retain 173 with complete timestamps and deterministically select 30 for manual audit, spanning 29 sessions and 24 repositories. The sample covers all nine mutation, task-notification, or tool-overlap events together with three assistant-only duration strata; it is intended for taxonomy construction, not prevalence estimation.

The audit identifies 20 revisions and reveals heterogeneous recovery structure: three affect a linear chain or the whole workflow (L0), ten primarily admit suffix recovery (L1), six exhibit an apparently independent branch (L2 candidate), and one has no useful intervention window. Crucially, apparent locality is not sufficient to authorize reuse. A branch that looks independent in the conversational trace may still depend on a revised artifact or control decision that the trace does not expose. SWE-chat lacks artifact versions and complete data/control provenance, so the L2 cases are only candidates for selective recovery rather than evidence that reuse is safe.

The workload study therefore establishes both sides of the motivation: revisions can overlap with ongoing agent execution, and some revisions appear local enough that coarse rollback may discard useful progress. At the same time, offline traces cannot certify which progress remains valid. This evidence gap motivates REVISE’s online provenance and fail-closed validity protocol.

## 3 REVISE

## 3.1 Validity and safe reuse

Intuitively, an earlier attempt is reusable only when the data it read, the control decision it followed, its parent work, and any effect it would publish all remain valid. For example, a revision to plan.budget invalidates an attempt that read that field, but need not invalidate an independent attempt that read neither that field nor an affected parent.

A workflow is a DAG $G = ( V , E )$ whose nodes may invoke models, tools, verifiers, routers, or joins. Each semantic artifact a has a monotonically increasing version $v _ { a } . \mathrm { A }$ read selector $( a , p , c )$ identifies an artifact, a nested semantic path, and whether the read supplied data or controlled execution $( c \in \{ \mathsf { d a t a } , \mathsf { c o n t r o l } \} )$ . A revision event $\boldsymbol { e } = ( a , v , v + 1 , o p , \Delta )$ replaces, appends, cancels, or changes control state; ∆ is the set of changed paths. The runtime accepts e only when v is current. An attempt $x _ { u }$ is valid if no revision since its snapshot intersects any data or control selector it consumed, every parent attempt remains current, and its effect remains authorized. The core invariant is

![](images/3a9c37184ce250d5ee5817cf9310461ffd27bf314921d5fffa52a17937d9f26e.jpg)  
Figure 3: REVISE recovery from revision to commit. A revision arrives with completed, running, and pending work. REVISE intersects its version delta with dynamic data/control provenance, maps validity to lifecycle actions—reuse, cancel, continue, avoid, or recompute—and revalidates joined outputs before commit. Complete evidence enables selective recovery; partial or unknown evidence expands recovery to a suffix or full restart.

$$
\mathsf { r e u s e } ( x _ { u } ) \Rightarrow \mathsf { v a l i d } ( x _ { u } ) \wedge \mathsf { e f f e c t } { \mathsf { S a f e } } ( x _ { u } ) .\tag{1}
$$

When either condition cannot be proved, REVISE recomputes or blocks rather than guessing.

## 3.2 Dynamic impact and recovery

REVISE first identifies the work that a revision can affect. It finds recorded reads that intersect the changed paths ∆. Ordinary paths intersect under ancestor–descendant overlap. A structural read uses a #members marker to record a collection’s membership, so member insertion or deletion intersects the marker only for that collection. To collect reads without manual dependency labels, an adapter wraps structured state at node entry in a recursive read-only proxy: field access records a leaf path, whereas iteration records a membership marker. The execution framework supplies parent edges. A typed patch is merged into the current snapshot, and an old/new structural diff produces $\Delta$ without an application-provided affected set.

REVISE indexes complete readers by artifact path, coarse readers by artifact, and unknown readers in a conservative set. For each revision, it finds direct readers whose selectors intersect $\Delta$ and traverses their active descendants. This produces an impact set and a recovery plan that records direct taint, affected nodes, unsafe effects, and the evidence behind every decision.

The plan assigns each node one of five lifecycle actions:

• Cancel: stop affected work that is currently running.

• Avoid: do not start affected work that is still pending.

• Recompute: execute again, on the revised snapshot, any affected completed node and any canceled or avoided node still required by the revised workflow.

• Continue: let a proven-unaffected running attempt proceed; result remains provisional until commit.

• Reuse: keep a proven-unaffected completed result; it too is revalidated at commit.

Thus, preserve is not a separate action: it means continuing unaffected running work or reusing unaffected completed work. The first three actions remove or redo stale work, while the last two retain only work supported by the available evidence.

The same evidence determines the scope of recomputation. Complete fine-grained provenance recomputes only the affected subgraph. If REVISE can identify only the earliest safe conflict, it recomputes a topological suffix; if it cannot identify a safe boundary, it performs a full restart. An irreversible or incompletely covered effect requires explicit handling or blocks execution. Selective recovery is therefore an optimization enabled by the validity semantics.

## 3.3 Commit-time revalidation

A revision may precede an attempt’s final read, so revision-time decisions cannot permanently authorize reuse. Each attempt runs on an immutable snapshot and accumulates a certificate

$$
C _ { x } = ( i d , v _ { s t a r t } , R _ { d a t a } , R _ { c o n t r o l } , P , m o d e ) ,\tag{2}
$$

where $v _ { \mathrm { s t a r t } }$ is the revision-journal version at attempt start, P identifies parent attempts and mode is complete, coarse, or unknown. Revisions append deltas to a journal. Revision and commit serialize on the same lock. At commit, REVISE checks every delta after $v _ { s t a r t }$ against the final certificate and verifies that every parent remains the current committed attempt. A conflicting, unknown, or obsolete attempt is rejected; tool effects remain staged until this check succeeds. If a later revision invalidates an already committed parent, REVISE invalidates its dependent outputs and any revocable or versioned effects before recomputing the current version.<sup>1</sup>

## 4 Evaluation

Experimental setup. We evaluate REVISE on LangGraph and LLMCompiler [9], repositorygrounded replay, adversarial tests, and multi-tenant serving. Both applications use local Qwen3-14B inference through SGLang. Serving experiments use 64 tenants and four H100 replicas at 1.4–2.6 workflows/s (0.7–1.3× a measured 2.0-workflow/s capacity).

Repository-grounded replay. We construct controlled replays from SWE-Review-Traj [10]. Of 100 audited trajectories, 51 request patch changes and 33 expose structured defect paths; we select five locally reproducible workloads. We preserve the repository, patch, reviewer evidence, and tests while controlling revision timing, so the replay grounds revision content and tool work rather than natural timing.

Runtime and baselines. Our Python adapters obtain LangGraph topology/state reads and LLM-Compiler Task dependencies, with SGLang serving [3]; they also support request aborts and staged effects. We comparefull restart, earliest-conflict dynamic suffix, and REVISE selective recovery; live experiments additionally wait for turn end. Suffix receives the same provenance as REVISE, isolating recovery granularity. We report correctness, calls, tokens, model-wall time, recomputation, 25-s SLO goodput, and p99 latency.

## 4.1 RQ1: Does REVISE preserve correctness?

Table 1 summarizes 15,939 executions. Every run matches its latest-version/full-restart oracle, with no stale output or committed effect. The 300-run adversarial matrix covers both revision/commit orders, late reads, consecutive revisions, unknown provenance, membership changes, and controlroute invalidation. Obsolete attempts never publish, while valid siblings remain reusable; all 800 validity-certificate checks complete in under 0.04 ms. These results establish in-process validity and effect safety, not distributed consensus or crash recovery.

Table 1: Correctness and effect safety across complementary settings.
<table><tr><td>Workload</td><td></td><td>Runs Oracle-equal ↑ Stale out. ↓ Stale eff. ↓</td><td></td></tr><tr><td>Online and integration matrices</td><td>171</td><td>171</td><td>0</td></tr><tr><td>LLMCompiler portability</td><td>288</td><td>288</td><td>0</td></tr><tr><td>Repository-grounded replay</td><td>180</td><td>180</td><td></td></tr><tr><td>Adversarial protocol</td><td>300</td><td>300</td><td></td></tr><tr><td>GPU serving pressure</td><td>15,000</td><td>15,000</td><td></td></tr><tr><td>Total</td><td>15,939</td><td>15,939</td><td>0</td></tr></table>

## 4.2 RQ2: How much recomputation is avoided?

On LangGraph, REVISE reduces model calls by 56.0% versus full restart and 43.6% versus dynamic suffix; model-wall time falls by 62.7% and 50.2%. Across 48 LLMCompiler workflows, calls fall by 40.6% and 31.3%, while tokens fall by 7.9% and 5.8%, respectively.

Because dynamic suffix uses the same provenance, these gains come from finer recovery rather than better dependency information. With unavailable fine-grained provenance, REVISE falls back conservatively: all 48 cases lacking fine-grained provenance recover equivalently to full restart.

Repository replay shows the same effect on real coding artifacts. Across 18 replay configurations, REVISE performs 12.94% less recomputation than suffix (95% CI: 7.77–17.70%) and discards 10.38% less completed work (95% CI: 2.74–18.81%). All 180 executions remain equivalent in final output, pytest, file/patch state, and visible effects.

## 4.3 RQ3: Do work savings become serving gains?

Work reduction need not shorten a single request’s critical path. LangGraph E2E p50 improves only 5.9% over suffix despite 50.2% less model-wall time, while repository replay changes E2E by −0.55% (95% CI: [−4.00, +3.84]%). We therefore evaluate whether saved work helps under replica contention. Across 15,000 workflows with 50% revised requests, REVISE reduces revisionto-correct-completion tokens by 13.26% (95% CI: 12.86–13.67%) and model-wall time by 14.70% (95% CI: 14.28–15.11%) relative to suffix. Table 2 shows little goodput gain near capacity but increasing benefit under contention. Thus, work reduction is stable across load, while its serving value is pressure-dependent.

Table 2: Serving change of REVISE relative to suffix. Brackets report 95% confidence intervals.
<table><tr><td>Load</td><td>Regime</td><td>SLO goodput ↑</td><td>p99 ↓</td></tr><tr><td>1.0×</td><td>Frozen capacity</td><td>-0.12%</td><td>-1.13%</td></tr><tr><td>1.1×</td><td>Boundary pressure</td><td>+3.07%</td><td>-4.42%</td></tr><tr><td>1.3×</td><td>Intentional overload</td><td>+5.43%</td><td>-13.80%</td></tr><tr><td>Pooled 1.0-1.3×</td><td></td><td>+2.79% [+1.23, +4.40]-6.45% [−10.11, -3.05]</td><td></td></tr></table>

## 5 Related work

Revision and rollback recovery. Revisable by Design rolls a streaming trace back to its earliest conflict [5]. Error-triggered alternatives include stepwise/Web rollback [11, 12], context–environment rewind [13], and DART’s dependency- and effect-aware checkpoint admission [14]. These systems roll back trajectories or checkpoints. In contrast, REVISE maps an external semantic revision to leaf-level validity decisions across node lifecycles, allowing it to preserve non-contiguous work when preservation is proved safe.

Foundations, substrates, and workload characterization. Self-adjusting computation tracks dependencies and propagates changes [15]; REVISE extends this perspective to partially executed agent DAGs with cancellation, staged effects, and revision/commit races. AIOS, SGLang, and Agentix schedule agent work [2, 3, 4]; PASTE overlaps execution [16]; Leyline edits KV spans [17]; and DeltaBox/Crab restore sandboxes [18, 19]. TraceLab characterizes long-running coding-agent trajectories and human interaction gaps [20], but does not identify whether a later interaction revises already-consumed execution state or which completed or in-flight work remains valid afterward. ACRFence exposes replay hazards [21], while Atomix/Cordon protect externally visible effects [6, 7]. These systems provide complementary execution and recovery substrates, but do not jointly map online revisions to data/control-validity decisions over a partially executed workflow.

## 6 Conclusion

Agent workflows must remain correct when user revisions arrive during ongoing execution without unnecessarily discarding valid progress. We presented REVISE, a validity-guided runtime for finegrained recovery that identifies affected work from revision deltas and dynamic data and control dependencies, stops invalid work, selectively preserves proven-valid progress, and recomputes only the affected region. When provenance is incomplete, REVISE conservatively expands recovery toward a suffix or full restart, while outputs and staged tool effects are revalidated before commit. Analysis of real SWE-chat traces establishes opportunities for online intervention, and experiments on unmodified LangGraph and LLMCompiler applications, repository-grounded replays, challenging event orderings, and Qwen3-14B serving-pressure settings show that REVISE maintains latest-version correctness without stale outputs or effects, reduces unnecessary recomputation, and improves serving efficiency under contention.

## References

[1] Joachim Baumann, Vishakh Padmakumar, Xiang Li, John Yang, Diyi Yang, and Sanmi Koyejo. SWE-chat: Coding agent interactions from real users in the wild. arXiv preprint arXiv:2604.20779, 2026. URL https://arxiv.org/abs/2604.20779.

[2] Kai Mei, Xi Zhu, Wujiang Xu, Mingyu Jin, Wenyue Hua, Zelong Li, Shuyuan Xu, Ruosong Ye, Yingqiang Ge, and Yongfeng Zhang. AIOS: LLM agent operating system. In Conference on Language Modeling (COLM), 2025. URL https://openreview.net/forum?id=L4HHkCDz2x.

[3] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Efficient execution of structured language model programs. arXiv preprint arXiv:2312.07104, 2024. URL https: //arxiv.org/abs/2312.07104.

[4] Michael Luo, Xiaoxiang Shi, Colin Cai, Tianjun Zhang, Justin Wong, Yichuan Wang, Chi Wang, Yanping Huang, Zhifeng Chen, Joseph E. Gonzalez, and Ion Stoica. Agentix: An efficient serving engine for LLM agents as general programs. In 23rd USENIX Symposium on Networked Systems Design and Implementation (NSDI 26), pages 2443–2459. USENIX Association, 2026. URL https://www.usenix. org/conference/nsdi26/presentation/luo.

[5] Zhiyuan Zhai, Ming Li, and Xin Wang. Revisable by design: A theory of streaming LLM agent execution. arXiv preprint arXiv:2604.23283, 2026. URL https://arxiv.org/abs/2604.23283.

[6] Bardia Mohammadi, Nearchos Potamitis, Lars Klein, Akhil Arora, and Laurent Bindschaedler. Atomix: Timely, transactional tool use for reliable agentic workflows. arXiv preprint arXiv:2602.14849, 2026. URL https://arxiv.org/abs/2602.14849.

[7] Zheng Chen, Hanqing Liu, Duling Xu, Dong Dong, Jialin Li, Bangzheng Pu, and Jidong Zhai. Cordon: Semantic transactions for tool-using LLM agents. arXiv preprint arXiv:2606.17573, 2026. URL https: //arxiv.org/abs/2606.17573.

[8] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=VTF8yNQM66.

[9] Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael Mahoney, Kurt Keutzer, and Amir Gholami. An llm compiler for parallel function calling. arXiv preprint arXiv:2312.04511, 2023.

[10] Ruoyu Wang, Jierun Chen, Shaowei Wang, Chaofan Tao, Sidi Yang, Yuxin Jiang, Kim-Hui Yap, Lifeng Shang, Xiaohui Li, and Haoli Bai. SWE-Review: Closing the loop on issue resolution with agentic code review. arXiv preprint arXiv:2607.06065, 2026. URL https://arxiv.org/abs/2607.06065.

[11] Xingzuo Li, Kehai Chen, Yunfei Long, Xuefeng Bai, Yong Xu, and Min Zhang. Generator-assistant stepwise rollback framework for large language model agent. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 17683–17700, 2025. doi: 10.18653/v1/2025. emnlp-main.892. URL https://aclanthology.org/2025.emnlp-main.892/.

[12] Zhisong Zhang, Tianqing Fang, Kaixin Ma, Wenhao Yu, Hongming Zhang, Haitao Mi, and Dong Yu. WebRollback: Enhancing web agents with explicit rollback mechanisms. In Proceedings of the 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 187–197, 2026. doi: 10.18653/v1/2026.eacl-short.12. URL https://aclanthology. org/2026.eacl-short.12/.

[13] Yu Zhuang, Kefei Chen, Yitong Duan, Shuxin Zheng, Jian Li, and Xu-Yao Zhang. AgentRewind: Recoverable execution for long-horizon LLM agents. arXiv preprint arXiv:2608.14380, 2026. URL https://arxiv.org/abs/2608.14380.

[14] Ke Yang, Panpan Li, Zonghan Wu, Kejin Xu, Huaxi Huang, and Xiaoshui Huang. DART: Semantic recoverability for structured tool agents. arXiv preprint arXiv:2605.23311, 2026. URL https://arxiv. org/abs/2605.23311.

[15] Daniel Anderson, Guy E. Blelloch, Anubhav Baweja, and Umut A. Acar. Efficient parallel self-adjusting computation. In ACM Symposium on Parallelism in Algorithms and Architectures (SPAA), pages 59–70, 2021. doi: 10.1145/3409964.3461799.

[16] Yifan Sui, Han Zhao, Rui Ma, Zhiyuan He, Hao Wang, Jianxun Li, Kaiqiang Xu, Kai Chen, and Yuqing Yang. Parallelizing tool execution and LLM generation for low-latency agent serving. arXiv preprint arXiv:2603.18897, 2026. URL https://arxiv.org/abs/2603.18897.

[17] Bole Ma, Jan Eitzinger, and Harald K"ostler. Leyline: KV cache directives for agentic inference. arXiv preprint arXiv:2606.01065, 2026. URL https://arxiv.org/abs/2606.01065.

[18] Yunpeng Dong, Jingkai He, Shiqi Liu, Yuze Hou, Dong Du, Zhonghu Xu, Si Yu, Baochuan Yang, Yubin Xia, and Haibo Chen. DeltaBox: Scaling stateful AI agents with millisecond-level sandbox checkpoint/rollback. arXiv preprint arXiv:2605.22781, 2026. URL https://arxiv.org/abs/2605.22781.

[19] Tianyuan Wu, Chaokun Chang, Lunxi Cao, Wei Gao, and Wei Wang. Crab: A semantics-aware checkpoint/restore runtime for agent sandboxes. arXiv preprint arXiv:2604.28138, 2026. URL https://arxiv.org/abs/2604.28138.

[20] Kan Zhu, Mathew Jacob, Chenxi Ma, Yi Pan, Stephanie Wang, Arvind Krishnamurthy, and Baris Kasikci. TraceLab: Characterizing coding agent workloads for LLM serving. arXiv preprint arXiv:2606.30560, 2026. URL https://arxiv.org/abs/2606.30560.

[21] Yusheng Zheng, Yiwei Yang, Wei Zhang, and Andi Quinn. ACRFence: Preventing semantic rollback attacks in agent checkpoint-restore. In CoDAIM Workshop, 2026. URL https://arxiv.org/abs/2603. 20625.

## A Additional Experimental Details

## A.1 Online revision opportunity in SWE-chat

The 118 SWE-chat sessions with observable work before revision delivery contain 174 work-bearing queued-revision events. We retain 173 events from 117 sessions with complete enqueue and delivery timestamps.

Audit-cohort construction. The audit includes all nine rare strict-overlap events—one mutation, two task notifications, and six tool overlaps—plus seven assistant-only events from each duration stratum: < 10 s, 10–30 s, and ≥ 30 s. Within each stratum, a stable hash fixes the order and a greedy rule favors unused sessions and underrepresented repositories. The resulting 30-event cohort spans 29 sessions and 24 repositories. It is frozen before semantic review and is used for taxonomy construction, not prevalence estimation.

## A.2 Latest-version correctness

We evaluate latest-version correctness under challenging event orderings and repository-grounded recovery scenarios.

Table 3: Challenging event orderings and required recovery behavior.
<table><tr><td>Case</td><td>Required behavior</td></tr><tr><td>Late read</td><td>A read materialized after a revision is revalidated at commit; an invalid attempt is rejected.</td></tr><tr><td>Commit race</td><td>Commit-first output is subsequently invalidated; revision-first rejects the stale attempt at commit.</td></tr><tr><td></td><td>Consecutive revisions Obsolete v1/v2 attempts reject while a disjoint sibling remains valid.</td></tr><tr><td></td><td>Unknown provenance Missing dependency evidence fails closed.</td></tr><tr><td>Membership change</td><td>A #members read intersects key insertion or deletion.</td></tr><tr><td>Control route</td><td>Old route certificates invalidate and the current route recomputes.</td></tr></table>

Challenging event orderings. Both possible orderings between revision and commit are exercised in the 300-run matrix. Across all runs, REVISE matches the latest-version oracle without stale outputs or effects. Across 800 commit checks, certificate validation costs 0.0026 ms p50, 0.0076 ms p95, and 0.0344 ms maximum.

Table 4: Correctness on 30 reviewer-triggered issues from 20 repositories.
<table><tr><td>Policy</td><td></td><td>Official verdict ↑ Latest-version equivalent ↑</td><td>Stale effects ↓</td></tr><tr><td>Full restart</td><td>30/30</td><td>30/30</td><td>0</td></tr><tr><td>Earliest-conflict suffix</td><td>30/30</td><td>30/30</td><td>0</td></tr><tr><td>REVISE</td><td>30/30</td><td>30/30</td><td>0</td></tr></table>

Repository-grounded correctness. The official verdict is joined to a reference Patch@v2, and a three-task canary reruns the official evaluator under all three policies. This experiment tests recovery correctness rather than patch quality. A separate local Qwen3-14B canary resolves 0/3 tasks, so we make no claim that REVISE improves patch generation.

## A.3 Selective recovery efficiency

We evaluate how selective recovery reduces recomputation across LangGraph, LLMCompiler, and repository-grounded replay workloads.

LangGraph live-request results. The matrix delivers typed revisions during live SGLang generations and issues 15 real aborts. Policy and scenario order rotate across three repeats. Because the shared service cache is not flushed between jobs, calls and generated tokens are primary metrics; model-wall time is secondary.

Table 5: Qwen3-14B work and latency in the unmodified 10-node LangGraph application.
<table><tr><td>Policy</td><td>Calls ↓ Compl. tok. ↓ Model-wall ↓ E2E p50 ↓</td><td></td><td></td><td></td></tr><tr><td>Full restart</td><td>10.0</td><td>80.0</td><td>1521 ms</td><td>719 ms</td></tr><tr><td>Earliest-conflict suffix</td><td>7.8</td><td>62.4</td><td>1138 ms</td><td>711 ms</td></tr><tr><td>REVISE</td><td>4.4</td><td>35.2</td><td>567 ms</td><td>669 ms</td></tr></table>

Table 6: REVISE reduction relative to earliest-conflict suffix in the live-request Qwen matrix (means over three repeats).
<table><tr><td>Revision</td><td>Calls</td><td>Prompt tok.</td><td>Compl. tok.</td><td>Model-wall</td></tr><tr><td>Policy/control</td><td>37.5%</td><td>36.3%</td><td>42.6%</td><td>43.8%</td></tr><tr><td>Helper goal</td><td>28.6%</td><td>28.1%</td><td>28.6%</td><td>28.3%</td></tr><tr><td>Reviewer evidence</td><td>16.7%</td><td>15.9%</td><td>16.7%</td><td>16.8%</td></tr></table>

LLMCompiler portability. The portability matrix fixes eight ParallelQA plans, four revision classes, and three repeats per policy. All 48 local identities remain selective, whereas all 48 cases lacking fine-grained provenance recover equivalently to full restart. The resulting 50% fallback rate is a controlled stress-matrix composition, not a deployment estimate. All 288 runs match the latest-version oracle with no stale outputs or effects.

A 30-pair no-revision probe produces identical outputs. Its measured −4.2% wall-time difference is treated as scheduling noise rather than a speedup. Planner latency is 0.207 ms p50.

Repository-grounded locality. Aggregating the first two local cases yields confidence intervals excluding zero. The three ratio-one controls have intervals crossing zero and are therefore interpreted as policy equivalence rather than performance gains. These controls distinguish the benefit of preserving proven-valid progress under local revisions from cases in which fine-grained recovery correctly collapses to the suffix boundary.

## A.4 Robustness to provenance and revision frequency

The benefit of selective recovery depends on both how often revisions occur and how precisely the runtime can establish execution validity. We vary these two factors independently.

Revision-frequency sensitivity. Across 24,000 workflows with 0/10/25/50% revised requests, no-revision model-wall time changes by only +0.013%, with a confidence interval crossing zero. At nonzero revision fractions, per-revision completion-token and model-wall reductions remain approximately 12–14%. Aggregate benefit therefore increases with the controlled revision fraction; this sweep does not estimate production prevalence.

Provenance sensitivity. A separate 12,000-workflow provenance sweep yields recomputation-toactive-work ratios of 50.0%, 81.7%, and 100% under complete, half, and unknown provenance. Half coverage falls back to full-equivalent recovery for 60% of revised requests, while unknown provenance does so for 100%. All runs match the latest-version oracle with no stale effects. The half-coverage mask is controlled rather than a measurement of natural provenance coverage.

In the unmodified 10-node LangGraph application, complete provenance captures all six expected leaf reads and recovers all five expected recomputation sets without full-restart fallback. As provenance becomes less precise, recovery expands conservatively rather than reusing work whose validity cannot be established.

## A.5 Serving under GPU pressure

The serving matrix runs Qwen3-14B with SGLang 0.5.13 on four single-H100 replicas and 64 tenant namespaces under open-loop arrivals. A separate suffix-only scan fixes measured capacity at 2.0 workflows/s before policy comparison. We evaluate 1.4, 1.8, 2.0, 2.2, and 2.6 workflows/s, corresponding to 0.7–1.3× measured capacity; the final rate is intentional overload.

Table 7: Recovery work in the unmodified LLMCompiler application [9] (48 identities per scope; means).
<table><tr><td>Scope</td><td>Policy</td><td>Calls↓</td><td></td><td>Total tok. ↓ Full fallback</td></tr><tr><td>Local</td><td>Full restart</td><td>6.00</td><td>2536.5</td><td>100%</td></tr><tr><td>Local</td><td>Earliest-conflict suffix</td><td>5.19</td><td>2482.0</td><td>50%</td></tr><tr><td>Local</td><td>REVISE</td><td>3.56</td><td>2336.9</td><td>0%</td></tr><tr><td>Global/untracked All policies</td><td></td><td>6.00</td><td>2552.1</td><td>100%</td></tr></table>

Table 8: Selective recovery in the five repository-grounded replay workloads.
<table><tr><td>Role</td><td>Locality</td><td>Affected/suffix ratio</td><td>Recompute-work change</td></tr><tr><td>Multi-file local</td><td>Local positive</td><td>0.50</td><td>-13.50%</td></tr><tr><td>Small local</td><td>Local positive</td><td>0.75</td><td>-11.18%</td></tr><tr><td>Mixed boundary</td><td>Policy-equivalent</td><td>1.00</td><td>-2.37%</td></tr><tr><td>Global change</td><td>Negative control</td><td>1.00</td><td>-0.57%</td></tr><tr><td>Scope expansion </td><td>Negative control</td><td>1.00</td><td>-3.94%</td></tr></table>

Each policy runs 500 workflows per configuration (180 warmup, 280 steady, 40 cooldown) in three paired repeats, yielding 15,000 records. Job identity, arrival, tenant, revision, and replica assignment are paired, while policy order rotates across repeats. All runs use four replicas, satisfy telemetry and arrival-generator checks, and commit no stale effects. Work deltas and configuration-level serving deltas are bootstrapped separately.

## A.6 Implementation and reproducibility

Framework integration. The unmodified 10-node LangGraph application uses a 25-line shared state/node-factory wrapper that captures all six expected leaf reads, with no application-node edits or selector annotations. For LLMCompiler, a shared 343-logical-line adapter wraps native task arguments and dependencies with no application edits or selector annotations, capturing all 1,584 expected explicit selector instances across the portability matrix.

Environment and artifacts. Experiments use Python 3.12, LangGraph 1.2.9, SGLang 0.5.13, Qwen3-14B in bfloat16, and H100 80GB GPUs. Generation uses temperature zero and fixed experiment-order seeds. Formal runners emit per-job JSON records containing configuration, selected and executed nodes, token/model-wall metrics, effect state, and oracle checks; separate analyzers enforce frozen gates. Repository replay additionally records rootfs identity, test collection, and exit status. The anonymized artifact package includes the runners, analyzers, preregistrations, and environment instructions.