# SysPilot: Cross-Library Performance Autotuning via Library Performance Contracts

Yineng Yan yinengy@utexas.edu University of Texas at Austin Austin, TX, USA

Haowei Chen hwchen@cs.utexas.edu University of Texas at Austin Austin, TX, USA

Christopher J. Rossbach   
rossbach@cs.utexas.edu   
University of Texas at Austin   
Austin, TX, USA   
Microsoft   
USA

## Abstract

Modern scientific and machine-learning applications compose many independently developed libraries. When these libraries run together, their independently managed runtimes can oversubscribe shared resources: using the same CPU cores, fighting over GPUs, or contending for memory bandwidth. Avoiding such contention requires careful con figuration across library boundaries. Existing autotuners can search large configuration spaces, but they typically treat the target application or system as a black box. They do not directly investigate what resources are shared between libraries, which runtime signals correlate with interference, or how tuning actions should be coordinated across library boundaries. As a result, they are ineficient for finding optimal configurations to mitigate contention.

This paper presents SysPilot<sup>∗</sup>, a cross-library performance autotuning system based on Library Performance Contract (LPC). A Library Performance Contract captures a library’s resource demands, assumptions, controls, and observability signals in LPCL, a specification language for describing li brary resource requirements. SysPilot synthesizes contracts from documentation and code artifacts, composes contracts from the libraries used by an application, derives resourceconflict hypotheses, profiles the application to validate those hypotheses, and selects candidate configurations. This de sign makes the Library Performance Contract a reusable unit of performance knowledge: a contract for a library can be reused across many applications that depend on that library. We evaluate SysPilot on a benchmark suite of cross-library contention scenarios. The results show that SysPilot matches the performance of autotuners that rely on application-specific manual information while retaining cross-application reuse. Compared with general-purpose au totuners such as OpenTuner and SMAC3, SysPilot improves benchmark performance metrics by up to an additional 38% over the best configuration found by these autotuners, while using 3× fewer runs.

## CCS Concepts

• Software and its engineering → Software performance;   
Specification languages; Software libraries and repositories;   
Runtime environments.

## Keywords

performance autotuning, cross-library composition, resource contention, library performance contracts, specification languages, large language models (LLMs), profiling

## 1 Introduction

Modern high-performance applications are more often built by composing libraries rather than by implementing all computation and resource management in one codebase. For example, scientific and machine-learning workflows often combine numerical libraries, data-processing libraries, and accelerator runtimes, such as NumPy, SciPy, CuPy, PyTorch, TensorFlow, OpenBLAS, and Intel MKL [1, 16, 32, 36, 47, 49, 55]. This composition model allows applications to reuse optimized kernels and hardware-specific implementations. However, it also means that resource management is delegated to multiple independently designed library runtimes that may share and contend for diferent kinds of system resources.

CPU-parallel libraries such as OpenMP [10], OpenBLAS [55] and Intel MKL [49] create worker threads that consume CPU, cache, and memory. Data-processing libraries such as TensorFlow tf.data [31], Hugging Face datasets [26], and database systems [37, 39] create workers, queues, and prefetch bufers that consume memory, storage bandwidth, and network or object-store bandwidth. GPU libraries such as PyTorch [36], TensorFlow [1], cuDNN [9], cuBLAS [33], and DALI [34] allocate device memory, launch kernels, and issue host-device transfers that consume GPU compute, device memory capacity, memory bandwidth, and PCIe or NVLink bandwidth. Each library may expose knobs for its own resource usage, such as thread counts, batch sizes, queue depths, memory limits, or communication bufer sizes. These controls are local: they can change how one runtime consumes CPU, GPU, memory, or network resources, but they do not capture how that resource use overlaps with other runtimes in the same application. As a result, a configuration that is eficient for one library in isolation can still cause contention when several libraries are sharing the same resources.

Recent work on Virtual Library Contexts (VLCs) [51] identifies this as a fundamental challenge in library composition: many libraries assume they own all process resources or expose only process-level configuration, so composing them can over-allocate cores or devices and degrade performance. Mechanisms for controlling resources at library granularity are emerging. For example, VLCs provide such a mechanism: they can virtualize libraries and partition resources without modifying library code. However, a mechanism does not by itself detect which libraries interfere, which shared resources are oversubscribed, or which resource partition yields the optimal performance. Cross-library contention mitigation still needs a diagnosis and tuning layer.

Traditional autotuners formulate tuning as search over an application-defined configuration space using measured performance feedback [2, 3, 17, 18, 42, 43, 50]. They are effective when the search space is compact or when enough trials are available, but they do not represent why inde pendently managed libraries interfere. Recent LLM-assisted tuners [14, 25, 45] move beyond purely black-box search by reading documentation, web pages, or code artifacts to generate tuning advice or candidate configurations. In Section 2, our motivating example shows that an LLM-assisted tuner given additional application/library descriptions could infer that the two tasks are LibTorch workloads competing for CPU cores and that the useful action is to search over asymmetric core partitions, rather than blindly exploring all per-core assignments. Such information greatly reduces the number of trials to find the optimal configuration by up to 6.4× compared with traditional autotuners.

However, existing LLM-assisted tuners still tend to organize knowledge around one target application or system [14, 25, 45]. This is helpful for a single workload, but it misses an important opportunity for reuse. In the LibTorch example, the exact workload split is workload-specific, but the stable part of the problem belongs to the library: LibTorch creates library-managed worker parallelism, consumes shared CPU compute, and exposes runtime signals that can reveal contention. An application-specific manual, prompt, or searchspace design must be rebuilt or reinterpreted for the next workload, even if that workload uses the same library and sufers from the same class of contention. Reusable librarylevel knowledge can instead transfer across applications that depend on the same library.

SysPilot is a novel solution for cross-library performance tuning based on this observation. It treats the library as the first-class application building block and captures a library’s performance characteristics using the Library Performance Contract Language (LPCL), a description language designed to capture the performance-relevant behavior of a library in a machine-readable form, including the resources the library uses, the controls that can change its behavior, and the signals that indicate performance degradation or contention.

Given LPCL for the libraries used by an application, SysPilot composes them to generate hypotheses about potential resource conflicts. Composition is a static analysis over contracts. The key idea is that LPCL gives each contract a common vocabulary for describing performance-relevant resources, controls, and signals. Once multiple contracts are expressed in this language, SysPilot can compare them systematically and identify overlapping resource declarations across libraries. These hypotheses are not treated as proof of slowdown. Instead, they determine which signals to collect, such as context-switch rate and memory footprint. Only hypotheses supported by profiling evidence generate corresponding candidate configurations. This hypothesis-guided workflow narrows the search space and ties each candidate configuration to an explicit diagnosis.

Organizing performance knowledge as library contracts makes that knowledge reusable: LPCL written once for popular libraries like OpenMP and PyTorch can be applied across many applications that depend on those libraries, rather than being rebuilt from scratch for each target application. In this sense, SysPilot shifts autotuning from an application-specific manual-building exercise to a cross-library reasoning problem over reusable library performance contracts.

An important aspect of this design is that LPCL is intended to be both authorable and generatable. A contract may be written directly by a library developer or performance expert, allowing human knowledge to be expressed in a structured, machine-readable form. To reduce the burden of manual authoring, SysPilot also includes an LLM-assisted automatic generation pipeline that harvests artifacts such as documentation, websites, configuration interfaces, and source code, extracts evidence from them, and synthesizes an LPCL contract. This generation pipeline is an important practical mechanism for obtaining contracts at scale.

We make the following contributions:

• We describe Library Performance Contracts as a reusable unit of performance knowledge for cross-library resource contention.

• We present LPCL, a specification language that describes a library’s performance characteristics, including the resource requirements, the controls that can change its resource usage, and the signals that indicate performance degradation.

• We design and implement SysPilot, an autotuner that synthesizes and checks LPCL, composes contracts into conflict hypotheses, profiles and collects signals, and searches for performant configurations.

• We build LibContentionBench, a benchmark suite that characterizes cross-library contention scenarios, and show that SysPilot achieves up to a 38% higher normalized score using 3× fewer trials than existing generic autotuners like OpenTuner and SMAC3 without writing oneof application-specific manuals to guide autotuning.

## 2 Motivation

Composing libraries that share the same resources may lead to resource contention and performance degradation. For example, a C++ LibTorch hyperparameter tuning task for machine learning models can be accelerated by parallelizing runs to leverage the large memory capacity and massive parallelism [27, 53]. When running two tasks in parallel, they contend for the same CPU resources. Because each task has diferent resource requirements, the optimal resource configuration is asymmetric: e.g., in a 24-core, two-socket NUMA system, the optimal configuration may assign 6 cores to the light task and 18 cores to the heavier task. The core assignment should also follow the NUMA topology.

For a general-purpose autotuner, however, finding this configuration is not trivial. If the tuner is given only a resourcepartitioning interface and end-to-end performance feedback, a natural encoding is to treat each of the 24 cores as an independent binary decision: assign this core to one of the tasks. This creates a $2 ^ { 2 4 }$ -point search space. This encoding hides the useful structure of the problem. The tuner is not really choosing among arbitrary per-core assignments; it is trying to find a good resource split between two competing library runtimes. A generic black-box autotuner does not know this distinction. It can only test candidate assignments and observe their performance, so it may spend many trials exploring configurations that are unlikely to resolve the underlying contention.

We use a simple experiment to illustrate how missing information has a direct efect on tuning eficiency. We implement a C++ LibTorch parallel hyperparameter tuning workflow on a transformer-based language model with 8 heads, 6 layers, and an embedding size of 512. The training dataset is wikitext2 [30]. We run two tuning tasks in parallel on a machine with two 12-core Intel E5-2650 v4 sockets, 128 GB of RAM. Each task can be configured with a variable number of CPU cores to avoid contention. We measure total training time for tuning a fixed number of hyperparameters.

Figure 1 shows the performance of the best configuration found so far by diferent autotuners. We compare SysPilot against two representative general-purpose autotuners:

![](images/515f3dfdf3fb23f112b8042da8c34c41e313432ef00cb52f1f399f3cfa1ea1f3.jpg)  
Figure 1: Best-so-far performance on the LibTorch hyperparameter tuning resource-partitioning example. The speedup is relative to the baseline that evenly splits cores between the two tasks. SysPilot finds the optimal configuration within 7 trials. OpenTuner and SMAC3 require 26 and 45 trials, respectively, to reach nearoptimal configurations. (Upper-left is better)

OpenTuner [2], an ensemble-based program autotuning framework that adaptively selects among multiple search techniques, and SMAC3 [18], a sequential model-based algorithm configuration framework based on Bayesian optimization. Because OpenTuner and SMAC3 treat candidate configurations largely as opaque points in a search space, they require 26 and 45 trials, respectively, to reach near-oracle configurations. In contrast, SysPilot uses library-level information to focus the search on contention-relevant resource partitions and reaches the best configuration within 7 candidate trials. The lesson is not merely that one optimizer is better than another. Rather, the example shows that the hard part is often not the numerical search procedure itself, but the absence of information that explains what resource is shared, which library controls afect that resource, and which class of actions is likely to relieve contention.

## 3 Related Work

Library resources and composition. The problem of composing parallel libraries has a long history. Lithe [35] proposes a common substrate that allows parallel runtimes to coordinate processors, but libraries must be ported to the Lithe interface. BOLT [20] reduces oversubscription for nested OpenMP parallelism by changing the OpenMP runtime. Lightweight Contexts [28] divide a process into separate resource and protection domains, but require kernel support. VLCs [51] provide a user-space mechanism for librarylevel virtualization inside a single process and can partition resources across libraries without modifying the libraries or the operating system. SysPilot is complementary to these mechanisms. It does not replace resource-management tools;

it decides which tool or conventional library knob is likely to address a validated cross-library conflict.

Autotuning and configuration search. Autotuners search configuration spaces using empirical feedback. OpenTuner [2] provides an extensible ensemble ofsearch techniques; Orio [17] uses annotations to generate and tune optimized code variants; ytopt [50] uses model-based optimization for scientific applications; and SMAC [18], Spearmint [42], Hyperopt [3], and Auto-WEKA [43] popularized Bayesian or model-based optimization for algorithm and hyperparameter configuration. FFTW [13] uses empirical planning to select fast transform implementations. These systems are efective when the search space is well defined, but they generally treat the target application as a black box. SysPilot instead uses a structured representation to shrink and organize the search space: it identifies plausible conflicts using LPCL, validates them with runtime signals, and uses validated hypotheses to narrow the configuration search space.

Prior work on tuning database systems shows that optimizers can benefit from representations richer than raw performance measurements. OtterTune [46] transfers tuning knowledge across database deployments; CDBTune [54] uses reinforcement learning for cloud database tuning; and DB-BERT [45], GPTuner [25], and �-Tune [14] use language models to extract or generate tuning guidance from textual artifacts. In networking, UNUM [7] uses a Transformer-based state embedder to learn latent representations of network conditions and improve network controllers such as con gestion control and adaptive bitrate streaming. These systems show that tuning and control can improve when the optimizer is given more informative representations than raw measurements alone. SysPilot adopts this lesson, but changes the unit of representation and reuse. Existing database tuners typically organize knowledge around one DBMS, deployment, or workload instance, while UNUM learns a representation for network-control state. SysPilot instead represents reusable library-level knowledge: a Library Performance Contract captures the resources, controls, signals, and validation rules of a library so that the same knowledge can transfer across applications that use that library.

Instance-specific optimization. A broader line of systems work argues that performance mechanisms should be specialized to the data, workload, hardware, or operating context of a particular deployment. The vision for instance-optimized data systems [24] argues that many data-system components, including query optimization, indexing, scheduling, and storage layout, can be specialized to a particular system instance. SysPilot adopts the same principle for cross-library tuning: library contracts provide reusable, general knowledge about resources and controls, but profiling is required to determine which contract-derived hypotheses actually hold for the current application. Thus, SysPilot combines reusable library-level knowledge with instance-specific validation to guide instance-specific tuning.

![](images/f4ec6056dde94e9b2581703c81bb0a5cec00fb8e1850de60066452125bc6656c.jpg)  
Figure 2: SysPilot architecture.

## 4 Design

SysPilot is a cross-library autotuning framework that turns library-level performance knowledge into a reusable Library Performance Contract (LPC) and uses it to guide autotuning. Figure 2 shows the architecture of SysPilot.

A Library Performance Contract describes how a library uses shared resources, which assumptions its runtime makes, which control knobs can change its resource demand and behavior, and which runtime signals indicate potential resource contention and performance degradation. Library developers or performance experts can write contracts in the domain-specific language LPCL described in Section 4.2. To reduce user burden, SysPilot includes an LLM-assisted contract generation pipeline that turns library documentation, code, and API artifacts into LPCL. At runtime, SysPilot composes the LPCL for the libraries used by the application, derives contention hypotheses, profiles the workload to val idate those hypotheses, and searches only over candidate configurations that are tied to validated conflicts.

## 4.1 Library Performance Contracts

A Library Performance Contract is a specification that characterizes performance-sensitive aspects of a library such as resource use, assumptions, control knobs, and performance metrics. A Library Performance Contract can be written in the Library Performance Contract Language (LPCL), a domain-specific description language as a collection of declarations including:

Components. Components are library-internal mechanisms that may consume or regulate resources. Examples include executors, queues, allocators, and backends. Components are not shared resources by themselves. For example, an Open-BLAS worker pool is an executor component; it becomes relevant to contention because it uses the shared resource cpu.compute.

Uses. A use declaration states that a component consumes or constrains a shared resource. For example, a BLAS executor uses cpu.compute with a worker demand in [1, host.cpu\_slots] Use declarations are the main input to contract composition because they make resource overlap explicit.

Assumptions. Assumptions describe conditions under which the library’s local resource policy is expected to behave well. Examples include exclusive for a workspace that assumes no competitors. Assumptions are useful because many crosslibrary failures occur when two individually reasonable local policies violate each other’s assumptions.

Controls. Controls are knobs, APIs, environment variables, and command-line flags that change a component attribute or a resource budget.

Signals. Signals describe what to measure and what the measurement indicates. The probe field is the string name of a virtual probe that will be resolved at runtime to a profiling script from SysPilot’s profiling toolbox, which is discussed in Section 5. Examples include active thread count, contextswitch rate, CPU utilization, GPU memory use, and cgroup memory events. Signals allow static contract composition to be checked against runtime evidence. A hypothesis is not considered active merely because two contracts mention the same resource; SysPilot collects the relevant signals before spending trials on candidate actions.

## 4.2 LPCL

Figure 3 shows the context-free grammar of LPCL. A contract begins with a named contract block and contains a sequence of declarations. The declaration types correspond to the contract concepts above: component, use, assume, control, signal. This structure is intentionally small. The goal is not to encode every detail of a library implementation, but to capture enough typed information for SysPilot to generate and validate hypotheses about cross-library resource contention.

LPCL has a fixed resource vocabulary. A use declaration must refer to one of the shared contention resources in Table 1. LPCL deliberately does not allow arbitrary resource names. For example, cpu.thread, worker.pool, and gpu.stream are rejected by the LPCL checker because they are not part of the resource vocabulary. This shared resource vocabulary allows contracts from diferent libraries to compose: two libraries may use diferent internal mechanisms, but they can still be compared through the same resource vocabulary.

The Demand fields are resource-specific: CPU compute uses worker counts, CPU placement uses core sets, GPU memory uses bytes, and I/O queue resources use queue depth.

A control has a target and an efect, such as setting a worker count, decreasing a queue depth, partitioning a resource, or disabling a cache. A signal has a target and a probe. For example, a thread count signal may indicate cpu.compute . pressure.

Finally, LPCL includes a metadata field evidence. The evidence field records where the declaration came from, such as a documentation page, API reference, or source-code fragment. It has no efect on composition or autotuning. This field is especially important for LLM-assisted contract generation because it makes generated contracts checkable and auditable rather than free-form advice.

## 4.3 Contract Composition and Hypothesis Generation

Given contract instances $C _ { 1 } , \ldots , C _ { n } ,$ SysPilot builds a resource ledger. Each ledger entry records a component, resource, demand interval, assumptions, controls, and signals. The composition algorithm applies a small set of conservative rules. Each rule emits a conflict hypothesis.

Budget violation. The sum of upper-bound demands can exceed a known budget. For CPU workers, this rule detects the common case where two libraries can each create up to host.cpu\_slots workers, which may lead to oversubscription.

Assumption violation. One contract states not\_nested, exclusive, or competitors at\_most 0 for a resource and another contract uses that resource in an overlapping phase. Placement conflict. Two entries constrain cpu.placement to overlapping CPU sets. Placement conflicts can also occur when one library pins workers and another assumes unconstrained scheduling.

� ::= contract ���� { �<sup>∗</sup> }   
� ::= ��������� | ��� | ������ | ������� | ��<sub>�</sub>���   
��������� ::= component ���� : ���� { evidence= ��������; }   
�<sub>��</sub> ::= use ���� { by= ���������; resource= ��������; ������; evidence= ��������; }   
�<sub>�����</sub> ::= assume ���� { resource= ��������; ���������; evidence= ��������; }   
�<sub>��</sub>�<sub>��</sub>� ::= control ���� { target= ���<sub>�</sub>��; effect= ������; evidence= ��������; }   
��<sub>���</sub>� ::= signal ���� { target= ���<sub>�</sub>��; probe= ���������; evidence= ��������; }   
�<sub>�������</sub> ::= cpu.compute | cpu.placement | gpu.compute | gpu.memory | gpu.bandwidth |   
mem.capacity | mem.bandwidth | io.bandwidth | io.queue   
���� ::= executor | queue | allocator | backend   
�<sub>����</sub>� ::= Demand(cpu.compute) | Demand(cpu.placement) |   
Demand(gpu.compute) | Demand(gpu.memory) | Demand(gpu.bandwidth) |   
Demand(mem.capacity) | Demand(mem.bandwidth) | Demand(io.bandwidth) | Demand(io.queue)   
Demand(cpu.compute) ::= workers in [����, ����] | workers= ����   
Demand(cpu.placement) ::= cores subset\_of ������ | unconstrained   
Demand(gpu.compute) ::= devices in [����, ����]   
Demand(gpu.memory) ::= bytes in [����, ����]   
Demand(gpu.bandwidth) ::= bandwidth= ����   
Demand(mem.capacity) ::= bytes in [����, ����]   
Demand(mem.bandwidth) ::= bandwidth= ����   
Demand(io.bandwidth) ::= bandwidth= ����   
Demand(io.queue) ::= depth in [����, ����]   
�<sub>��</sub>��<sub>��</sub>�<sub>�</sub> ::= not\_nested | exclusive   
<sup>�</sup>��<sub>�</sub>�� ::= ��������� | � �� | ��������   
� � � ��� ::= set ���<sub>�</sub>�� to ���� | decrease ���<sub>�</sub>�� | partition �������� | disable ���<sub>�</sub>��  
Figure 3: LPCL context-free grammar. The grammar contains the five declaration types used in this paper: component, use, assume, control, and signal. For signal, the probe field names a virtual probe in SysPilot’s profiling toolbox. The resource-specific demand forms define how each resource type expresses demand. The evidence field records raw evidence related to this declaration for user auditing purposes and is not used by the runtime.

For example, consider a workload that uses both OpenMP and OpenBLAS. These two libraries expose diferent controls and have diferent internal executor components, but both components consume cpu.compute. By mapping both uses into the same resource ledger, SysPilot can derive a reusable oversubscription hypothesis. The generated hypothesis will specify that SysPilot should collect context-switch rate and CPU utilization. It may list related controls such as changing an environment variable that controls the BLAS thread count.

## 4.4 Hypothesis-guided Autotuning

## SysPilot’s runtime loop has four phases.

Baseline profiling. The runtime launches the target command, profiles the process tree, collects signals from all startup hypothesis profiles, and records the objective metric. The default objective is wall time, but the user may specify a workload metric such as throughput.

Hypothesis validation. For each hypothesis, SysPilot collects observed signals. For example, a CPU oversubscription hypothesis may require a high runnable-thread count and a high context-switch rate as signals. If the signals are present, the hypothesis is validated.

Candidate generation. Only validated hypotheses produce candidate configurations. LPCL control declarations define the legal values for each control. A candidate may coordinate multiple controls associated with one validated conflict. For example, an oversubscription candidate may jointly limit the inner OpenMP and OpenBLAS thread pools while adjusting outer concurrency.

Execution and selection. Candidate configurations run through the same profiled executor as the baseline. SysPilot accepts a candidate only if the configured objective improves. When multiple hypotheses are validated, SysPilot tries candidates from each hypothesis in a round-robin manner and stops when the given tuning budget is exhausted, the objective converges, or no more candidates are available.

## 5 Implementation

We implemented a prototype of SysPilot as a generic endto-end autotuning framework. The implementation has three

<table><tr><td>Resource</td><td>Meaning</td></tr><tr><td>cpu.compute</td><td>CPU execution capacity consumed by runnable workers, thread pools, and</td></tr><tr><td>cpu.placement</td><td>task executors. Affinity and core-placement con- straints over CPU sets and NUMA domains.</td></tr><tr><td>gpu.compute</td><td>GPU compute units consumed by ker- nels.</td></tr><tr><td>gpu.memory</td><td>GPU memory capacity consumed by buffers.</td></tr><tr><td>gpu.bandwidth</td><td>GPU memory traffic throughput from</td></tr><tr><td>mem.capacity</td><td>reads and writes. Host memory capacity consumed by</td></tr><tr><td>mem.bandwidth</td><td>buffers. Host memory traffic throughput from</td></tr><tr><td>io.bandwidth</td><td>reads and writes. Storage or network throughput.</td></tr><tr><td>io.queue</td><td>Outstanding I/O concurrency or re- quest queue depth.</td></tr></table>

Table 1: The LPCL resource vocabulary. The language can be extended with new resource vocabulary, but the initial set shown here focuses on common shared resources that can lead to cross-library contention.

layers: LPCL generation, contract composition, and hypothesisguided autotuning.

## 5.1 Automatic LPCL Generation

LPCL can be written by experts, but the prototype also includes an LLM-assisted generation pipeline to reduce human efort. This also makes SysPilot an end-to-end autotuning framework that does not require users to be systems or library experts. Generation is bounded by schemas and has four passes.

Ingestion and chunking. The generator searches for relevant information online and ingests local documentation, README files, source-code snippets, and web pages. It chunks text into small segments and indexes them by source IDs.

Evidence extraction. An extraction prompt asks the LLM to emit evidence records for components, resource uses, controls, signals, and assumptions. Each evidence item must cite source chunks and use only the fixed resource vocabulary. Unsupported statements are omitted rather than guessed.

Synthesis. A second prompt asks the LLM to synthesize an LPCL contract from evidence records. The prompt includes the LPCL schema and extracted evidence. The LLM must choose which evidence to include, how to map it to the schema, and how to resolve conflicts. The output is a

YAML intermediate representation that can be checked and rendered to LPCL via a deterministic compiler.

Repair and compilation. The LLM-generated LPCL will be checked against the language syntax and grammar. If validation fails, a bounded repair loop rewrites only invalid sections. Users can also manually check evidence blocks to validate that the LPCL generated by the LLM matches the facts. In our experience, the LLM is able to produce a valid LPCL in one run without human correction using recent LLM models such as GPT-5.4.

This pipeline uses LLMs where natural-language understanding is useful, but all runtime artifacts are structured and checked. At tuning time, SysPilot operates on validated contracts and hypotheses rather than raw LLM outputs.

## 5.2 Virtual Probes and Profiled Execution

Profiling is the step that turns a statically generated conflict hypothesis into workload-specific evidence. However, LPCL does not bind a signal directly to a profiling script. A direct binding would make contracts platform-specific: the script used to measure a context-switch rate on Linux/x86 may difer from the script used on another operating system, processor, accelerator stack, or cluster environment. It would also duplicate information across contracts, since diferent libraries may rely on the same runtime signal. For example, an OpenBLAS contract and a LibTorch contract may both need to observe thread pressure or context-switch behavior when validating a cpu.compute contention hypothesis.

To keep contracts independent from profiling implementations, SysPilot uses virtual probes. A virtual probe is a symbolic probe name. For example, a signal may request the virtual probes context\_switch\_rate and threads\_count. During hypothesis generation, SysPilot resolves each virtual probe name to a concrete collector in the SysPilot profiling toolbox. The toolbox contains common platform-specific collectors shipped with SysPilot, as well as user-provided or LLM-generated collectors. For example, on a Linux/x86 platform, the virtual probe context\_switch\_rate may be resolved to context\_switch\_rate\_linux\_x86.sh. If multiple contracts request the same virtual probe, they are mapped to the same collector and collected once for the run.

This indirection separates three concerns. The contract specifies what signal is useful for validating a library-level hypothesis. The profiling toolbox specifies how that signal is collected on a particular platform. The runtime decides whether to invoke each collector based on the active hypotheses for the current workload. As a result, the same LPCL contract can be reused across platforms, and the same profiling script can serve many libraries and hypotheses.

The profiled executor launches the workload command and starts the signal collectors required by the active hypothesis profiles. Probes can be one-shot or session-based. A one-shot probe records a snapshot, such as an afinity mask or a device-memory summary. A session-based probe runs alongside the workload and samples periodically, such as collecting thread counts, context-switch rates, GPU utilization, or I/O throughput over time. The executor records all measurements in a single run artifact so that baseline and candidate executions are comparable.

After the baseline run, SysPilot uses the collected probe outputs to validate hypotheses. Only hypotheses supported by the requested probe evidence generate candidate configurations. The same profiled executor is then used for candidate runs, allowing SysPilot to compare both the objective metric and the validation signals before accepting a configuration.

## 6 Evaluation

All experiments are performed on a machine with two 12- core Intel E5-2650 v4 sockets, 128 GB of RAM, and 4 NVIDIA Tesla P100-SXM2 GPUs, running Ubuntu 22.04 with kernel 5.15.0. The LLM model used in this evaluation is GPT-5.4 with medium thinking efort.

## 6.1 LibContentionBench

Evaluating cross-library contention requires a benchmark whose unit of evaluation is a library interaction, not a single kernel, algorithm, DBMS, or monolithic application. Existing benchmark suites cover important adjacent goals, but they are not organized around this problem. Architecture and heterogeneous-computing suites such as PARSEC [4] and Rodinia [6] study applications and kernels with diverse hardware behavior, but they do not expose independently managed libraries and their resource controls as first-class tuning targets. End-to-end ML, database, cloud, and latency critical benchmarks such as MLPerf [29], TPC-H [44], Cloud-Suite [12], and TailBench [23] provide realistic workload drivers and standardized metrics. These suites are valuable for comparing complete systems, but they do not make independently managed library runtimes, their resource assumptions, and their cross-library conflicts the object of evaluation. Autotuning and algorithm-configuration suites such as AClib [19] and HPOBench [11] provide standardized parameterized objectives for comparing optimizers. They are closer to the evaluation methodology we need, but their abstraction is a target algorithm or HPO objective function, not a workflow composed of library runtimes with independent resource assumptions.

We built LibContentionBench to fill this gap. Its nine benchmarks compose widely used ML, data-processing, and

HPC libraries and expose their library and runtime configuration knobs. It includes five synthetic-data applications, including Joblib/BLAS Nested CPU Contention, Dask/DuckDB Columnar Memory-Spill, Hugging Face / PyTorch Text Data Pipeline, TorchData/S3-DDP Object-Store Pipeline, and DALI / PyTorch GPU Decode-Inference, whose input data is generated and whose workload sizes are bounded so that repeated tuning runs are reproducible and afordable. It also includes four real-data applications: Forest-Cover Model Selection, Movie Recommendation Training, Scientific Claim Retrieval, and SIFT Visual Codebook Construction, to demonstrate the performance of SysPilot on real-world workloads.

At the same time, each workflow preserves a realistic contention pattern in which independently managed library runtimes compete for CPU cores, memory capacity, memory bandwidth, I/O bandwidth, object-store access, or GPU resources. Input data, workload sizes, and algorithmic settings remain fixed during each tuning session; only the exposed resource controls may change.

LibContentionBench is not intended to be exhaustive. Instead, it is designed to cover the main resource and control dimensions of cross-library contention. The nine workflows span CPU oversubscription, NUMA-sensitive CPU partitioning, host-memory and spill-path contention, datapipeline worker and queue contention, object-store input contention, distributed-training input coordination, and GPU decode/inference co-scheduling. This coverage lets us evaluate whether a tuning method can reason across CPU, memory, I/O, and GPU resources, rather than focusing on only one resource type or one familiar library. Appendix A provides a detailed description of each application.

## 6.2 Application-Level Manual Baseline

All tuners receive the same target command, legal tuning surface, primary objective, timeout policy, and correctness constraints. The known optimal configuration (oracle) is hidden from tuners and used only to normalize results. The primary budget is 30 candidate configuration trials; we report best-so-far quality after each candidate execution.

Because workflows have diferent baseline-to-oracle headroom, we normalize performance by the oracle gap. Let $S _ { w } ^ { b a s e }$ be the baseline score for workflow �, $S _ { w } ^ { o r a c l e }$ be the oracle score, and $S _ { m , w } ^ { b e s t } ( b )$ be method �’s best score after � candidate executions. We report

$$
Q _ { m , w } ( b ) = \frac { S _ { m , w } ^ { b e s t } ( b ) - S _ { w } ^ { b a s e } } { S _ { w } ^ { o r a c l e } - S _ { w } ^ { b a s e } } .\tag{1}
$$

For throughput and fair-share metrics, � is the measured score. For lower-is-better runtime metrics, we first convert runtime to a speedup-style score relative to the baseline. Thus $Q = 0$ is baseline quality and $Q = 1$ is oracle quality.

![](images/55569405e5e4adaa24649984d76fad3bd753360beb26ded3050248f9fbc211f1.jpg)  
Figure 4: Best-so-far normalized score over 30 candidate trials for the nine benchmarks in LibContentionBench. The score is normalized with respect to the baseline and the best observed reference across the four methods. SysPilot with library contracts converges faster than generic autotuners and reaches near-oracle quality within a small budget. (Upper left is better)

We compare four methods. SysPilot (Lib Contract-Guided) uses reusable library contracts and the runtime loop described in Section 4. SysPilot (App Manual-Guided) uses the same executor and candidate-validation machinery, but replaces reusable contracts with a per-workflow application manual that describes the application-specific information. This represents a strong application-aware baseline that adopts the idea of the application-specific LLM-assisted autotuners in database and storage systems [14, 25, 45] while ensuring fairness by using the same SysPilot tuning pipeline and the same LLM model. SMAC3 and OpenTuner receive the same legal search space and objective but no library contracts [2, 18].

## 6.3 End-to-end Tuning Results

Figure 4 shows the best-so-far normalized score over the tuning budget. SysPilot with library contracts reaches the 0.8 average target in 8 trials and the 0.9 target in 12 trials. The application-manual variant reaches these targets in 6 and 12 trials, respectively. Among the generic autotuners, SMAC3 reaches 0.8 only after 24 trials, while OpenTuner does not reach it within the budget; neither generic method reaches 0.9.

At 30 trials, contract-guided tuning achieves the highest average normalized score, 0.960, compared with 0.935 for the application-manual baseline, 0.824 for SMAC3, and 0.729 for OpenTuner (Table 2). Application manuals provide strong target-specific guidance, particularly for the synthetic-data workflows. These workflows have intentionally bounded workload structures and relatively simple, well-defined li brary interactions, making it feasible to write a manual that accurately captures the important resource usage patterns and tuning opportunities.

<table><tr><td rowspan="2">Tuner</td><td colspan="2">Trials to target ↓</td><td colspan="2">Avg. score ↑</td></tr><tr><td>@0.8</td><td>@0.9</td><td>@10 runs</td><td>@30 runs</td></tr><tr><td>SysPilot (App Manual-Guided)</td><td>6</td><td>12</td><td>0.896</td><td>0.935</td></tr><tr><td>SysPilot (Lib Contract-Guided)</td><td>8</td><td>12</td><td>0.879</td><td>0.960</td></tr><tr><td>SMAC3</td><td>24</td><td> $> 3 0$ </td><td>0.611</td><td>0.824</td></tr><tr><td>OpenTuner</td><td>&gt; 30</td><td> $> 3 0$ </td><td>0.625</td><td>0.729</td></tr></table>

Table 2: Aggregate convergence summary across the nine LibContentionBench benchmarks. Trials to target columns (lower is better) report the first candidate execution at which the average normalized score reaches the target; > 30 means the target is not reached within the 30-trial budget. Average score columns (higher is better) report the average normalized best-so-far score at fixed budgets.

This advantage becomes less consistent on the real-data applications, whose execution involves more complex interactions among libraries and runtime components. In such applications, it is dificult for a manually written description to capture every performance-relevant detail in advance, and omitted details can make the guidance less efective. Library contracts instead describe reusable performance behavior at the library level and compose this information for the application being tuned. As a result, SysPilot retains the benefits of structured performance knowledge while avoiding the need to construct a complete application-specific description for every workload. This allows contract-guided tuning to achieve comparable or better aggregate performance than application-manual guidance while substantially outperforming generic autotuners. Both of them outperform generic autotuners in terms of convergence speed and final quality.

## 6.4 Case Study: How SysPilot Narrows the Search Space

This section uses the Joblib/BLAS nested CPU contention benchmark as a case study to explain why SysPilot outperforms generic autotuners under the same 30-trial budget. The workload contains nested parallelism: Joblib launches multiple outer tasks, and each Joblib task performs NumPy matrix multiplication. NumPy dispatches this computationally intensive operation to the numerical runtime linked into its build, such as OpenBLAS, MKL, or BLIS. Depending on the build, additional thread-pool controls such as OpenMP may also apply. A configuration $( j , o , b , m , l , k )$ specifies Joblib workers, OpenMP, OpenBLAS, MKL, and BLIS thread limits, and the Joblib backend. Numeric controls range over [1, 48] and the backend has two choices, giving $2 \times 4 8 ^ { 5 } = 5 0 9 , 6 0 7$ ,936 requested tuples.

The executed NumPy build uses OpenBLAS. Runtime evidence identifies �, �, and � as inactive, so SysPilot fixes them to one and searches (�, 1, �, 1, 1, �): 48 × 48 × 2 = 4,608 tuples. Contracts then relate outer Joblib workers and inner BLAS capacity to the shared CPU resource. Generic tuners lack this applicability information and must use their observations both to identify influential controls and to optimize their values. Moreover, many configurations within this reduced space can be eliminated because their combinations of outer workers and inner BLAS threads oversubscribe the 48 physical cores. The reduction therefore removes two major sources of wasted trials.

Within 30 trials, contracts reach a normalized score of 0.927, compared with 0.706 for SMAC3 and 0.774 for Open-Tuner; the application manual reaches 1.0. The distinction is how the budget is spent: SysPilot focuses on Joblib concurrency, the active OpenBLAS pool, and the backend, rather than repeatedly varying controls that do not afect this execution.

## 6.5 Cross-application Manual Transfer

The previous experiment compares SysPilot with applicationlevel manuals that are written specifically for each workflow. Such manuals are useful as target-specific priors, but they do not by themselves demonstrate reusable performance knowledge. To evaluate whether an application manual can be reused as a tuning artifact, we run a cross-application manual-transfer experiment on three PyTorch-related workflows: the Hugging Face / PyTorch Text Data Pipeline, the TorchData/S3-DDP Object-Store Pipeline, and the DALI/Py-Torch GPU Decode-Inference workflow.

All three workflows contain PyTorch, but they compose PyTorch with diferent surrounding libraries and stress different contention resources. The text pipeline combines contention among dataset workers, PyTorch CPU threads, and DataLoader workers. The object-store pipeline involves contention over object-store I/O and PyTorch CPU threads. The GPU pipeline involves contention over shared GPU compute units and GPU memory. This makes the set a useful test of whether an application-level manual transfers across applications that share a major library family but difer in cross-library contention structure.

![](images/b017836e8a7c92039a8e85bdb65683ac14895a01a9148a42b37da57eaeb491f5.jpg)  
Figure 5: Cross-application manual transfer among three PyTorch-related workflows. Rows are source manuals and columns are target applications. Each cell reports the number of trials required to reach 98% of the oracle score. Diagonal cells use the target’s own manual and serve as the row baseline; of-diagonal cells reuse a diferent application’s manual without editing it. Values in parentheses show slowdown relative to the diagonal baseline for the same source manual. (Lighter color is better)

For each target workflow, every method receives the target command, objective, legal knob schema, and correctness constraints. The only transferred artifact is the natural-language application manual. A diagonal entry uses the target’s own manual and represents the fresh-manual baseline. An ofdiagonal entry uses a manual written for a diferent workflow without editing it for the target. We measure the number of candidate executions required to reach at least 98% of the oracle score. Lower values are better.

Figure 5 shows that application manuals are efective when written for the target but brittle when reused. With the matching manual, the HF/PyTorch text pipeline reaches the 98% threshold in 5 trials, the TorchData/S3-DDP pipeline in 9 trials, and the DALI/PyTorch GPU pipeline in 11 trials. Of-diagonal transfer is much less reliable: the HF/PyTorch manual requires 27 trials on the TorchData/S3-DDP target and 26 trials on the DALI/PyTorch target, corresponding to 5.4× and 5.2× slowdowns relative to its own fresh-manual baseline. The TorchData/S3-DDP manual requires 30 trials on the HF/PyTorch target, a 3.3× slowdown. The DALI/Py Torch manual requires 21 and 28 trials on the other two targets, corresponding to 1.9× and 2.5× slowdowns.

One transfer case remains close to its source baseline: using the TorchData/S3-DDP manual for the DALI/PyTorch target reaches the threshold in 11 trials, only 1.2× the TorchData manual’s own baseline. This exception is important because it shows that transfer is not forced to fail. Rather, it is unstable. A whole-application manual entangles stable library behavior, application-specific structure, and deployment-specific assumptions. Reusing it transfers all three together, including assumptions that may not hold for the new target. SysPilot instead factors the stable part into library contracts and composes only the contracts needed for the target workflow.

These results support the design choice in SysPilot. Application manuals entangle three kinds of knowledge: stable library behavior, application-specific call structure, and deployment-specific contention assumptions. Reusing the entire manual transfers all three together, including assumptions that may not hold for the new target. SysPilot instead factors the stable part into library-level contracts. Contracts for PyTorch, DALI, fsspec/S3, DataLoader, or DDP can be selected and composed for the target workflow, while profiling validates which resource-conflict hypotheses are active in the current run. Thus, SysPilot does not require a fresh application manual for each new workflow, and it avoids treating a previous application’s manual as a monolithic tuning prior.

Overall, the manual-transfer experiment shows that fresh application manuals are strong target-specific baselines, but they are poor reusable artifacts. Even within a small set of PyTorch-related applications, transferring a manual from one workflow to another can increase the trials needed to reach near-oracle performance by up to 5.4×. This motivates using composable library-level contracts as the reusable unit of tuning knowledge instead of whole-application manuals.

## 6.6 Ablation Study

We next investigate how diferent SysPilot features contribute to search-eficiency improvement. This ablation uses the same transformer hyperparameter search workload from the motivating example in Section 2: two concurrent C++ LibTorch transformer tasks run on a 24-core machine with two diferent hyperparameter settings, and the tuner must find the core allocation that maximizes performance.

We therefore compare four variants that difer in how much structured information is available to the tuner. The primary metric is the number of candidate trials required to find the oracle configuration; lower is better. For the knobonly baseline, the tuner is capped at 30 trials, so we report 30+ when it fails to find the oracle within the budget.

Table 3 shows that each source of structure contributes to search eficiency.

The full system. The full system uses the complete SysPilot pipeline with all features described in the paper enabled. It finds the oracle in 7 trials because LPCL identifies the relevant shared resource and candidate interventions, while profiling validates which contract-derived hypotheses are active in the current run.

Table 3: Ablation study on two concurrent C++ LibTorch transformer tasks. (Lower is better).
<table><tr><td>Variant</td><td>Trials to oracle</td></tr><tr><td>Full SysPilot</td><td>7</td></tr><tr><td>Without probes/profiling</td><td>10</td></tr><tr><td>Raw documentation instead of LPCL</td><td>28</td></tr><tr><td>Knob-only autotuner</td><td>30+</td></tr></table>

No Profiling. This variant keeps the LPCL contracts and the contract-composition logic, but removes virtual probes and runtime profiling. As a result, SysPilot can still generate hypotheses from the contracts, but it cannot check which hypotheses are actually active in the current workload. The tuner therefore treats all contract-derived hypotheses as plausible and generates candidates for a larger set of possible conflicts. This increases the number of trials to 10. The result shows that LPCL alone provides useful structure, but profiling further reduces the search space by filtering out hypotheses that are not supported by runtime evidence.

Raw Documentation without LPCL. This variant replaces LPCL contracts with raw natural-language documentation about LibTorch, while keeping the same tuning harness and knob schema. The tuner can read relevant text, but the information is not represented as typed resources, controls, signals, and assumptions. Therefore, the system cannot deterministically compose contracts, build a resource ledger, or translate library knowledge into a small set of resourceconflict hypotheses. Candidate generation relies on free-form interpretation of the documentation, which is much less precise. This variant requires 28 trials to find the oracle. The result shows that the benefit is not simply from giving the tuner more text; the information must be structured so that it can guide hypothesis generation and narrow the root cause of contention.

Knob-only autotuner baseline. This baseline receives only the legal tuning-knob schema, the objective, and the same execution harness. It receives no LPCL contracts, no raw documentation, and no library-level explanation of what the knobs control. Consequently, the tuner has no information about which libraries are competing, which resource is shared, or which actions are likely to relieve contention. The search degenerates into black-box exploration over a large combinatorial configuration space. This baseline does not find the oracle within the 30-trial budget. The result shows that the improvement does not come from the autotuning framework alone; it comes from the library-level information supplied through LPCL and refined through profiling.

Overall, the ablation study shows that the improvement does not come from the autotuning framework alone. The knob-only baseline uses the same tuning interface but receives no library-level information, and it fails to find the oracle within the trial budget. Adding LPCL changes the problem from black-box knob exploration into hypothesisguided tuning: the contract identifies the relevant resources, controls, and contention mechanisms, allowing SysPilot to generate a small set of candidate actions targeted at the likely root cause. The raw-documentation variant shows that simply giving the tuner more text is not enough. LPCL improves over raw LibTorch documentation because it represents library knowledge in a structured, composable form that can be checked, combined, and translated into resource-conflict hypotheses. Profiling then provides a second reduction in search space by validating which contract-derived hypotheses are active in the current workload, preventing the tuner from spending trials on plausible but irrelevant interventions.

## 7 Limitations

Contract quality. SysPilot is only as good as its contracts. The quality of LLM-generated contracts depends largely on the available documentation. Documentation may omit performance behavior, source code may be hard to interpret, and an LLM extractor can miss or misclassify evidence.

Static false positives and false negatives. Contract composition is conservative. It may flag resources that are never used concurrently, or miss interference caused by an undocumented resource. Profiling is intended to filter false positives, but false negatives require better contracts or broader probes.

Version and build sensitivity. Libraries change behavior across versions, build options, and backends. The current prototype records evidence but does not yet provide full version-aware retrieval, versioned compatibility checks, or automatic contract aging.

## 8 Future Work

Several extensions follow naturally.

Version-aware contracts. Future contracts should include version ranges, backend conditions, build flags, and source provenance so the store can select the right contract for a particular deployment.

Distributed workflow tuning. LPCL intentionally uses a small vocabulary, and the main focus is on single-node tuning. However, the same principles apply to distributed jobs, where contracts can describe cluster resources, network links, and inter-node controls. Future work can extend the language and prototype to support distributed workflows.

## 9 Conclusion

Modern applications increasingly compose independently managed libraries, making shared-resource contention a cross-library problem rather than a single-runtime tuning problem. This paper presented SysPilot, a cross-library autotuning framework that uses Library Performance Contracts to make library-level performance knowledge reusable. We designed LPCL, a specification language that captures a library’s resource demands, assumptions, controls, and runtime signals. SysPilot synthesizes and checks these contracts, composes them into resource-conflict hypotheses, validates the hypotheses through profiling, and uses them to generate candidate configurations.

We evaluated SysPilot on the nine benchmarks in LibContentionBench. SysPilot with LPCL finds better configurations in fewer trials than general-purpose autotuners, achieving an average normalized score of 0.960, compared with 0.824 for SMAC3 and 0.729 for OpenTuner. SysPilot also reaches the 0.8 normalized score threshold in 3× fewer trials.

## References

[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jefrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geofrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 2015. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.

[2] Jason Ansel, Shoaib Kamil, Kalyan Veeramachaneni, Jonathan Ragan-Kelley, Jefrey Bosboom, Una-May O’Reilly, and Saman Amarasinghe. 2014. OpenTuner: An Extensible Framework for Program Autotuning. In Proceedings of the 23rd International Conference on Parallel Architectures and Compilation Techniques (PACT ’14). ACM, 303–316. doi:10.1145/2628071.2628092

[3] James Bergstra, Daniel Yamins, and David Cox. 2013. Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures. In International conference on machine learning. PMLR, 115–123.

[4] Christian Bienia, Sanjeev Kumar, Jaswinder Pal Singh, and Kai Li. 2008. The PARSEC benchmark suite: Characterization and architectural implications. In Proceedings of the 17th international conference on Parallel architectures and compilation techniques. 72–81.

[5] Jock A. Blackard and Denis J. Dean. 1998. Covertype Data Set. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/31/ covertype

[6] Shuai Che, Michael Boyer, Jiayuan Meng, David Tarjan, Jeremy W Sheafer, Sang-Ha Lee, and Kevin Skadron. 2009. Rodinia: A bench mark suite for heterogeneous computing. In 2009 IEEE international symposium on workload characterization (IISWC). Ieee, 44–54.

[7] Jiayi Chen, Nihal Sharma, Debajit Chakraborty, Saurabh Agarwal, Jefrey Zhou, Aditya Akella, and Sanjay Shakkottai. 2026. UNUM: A New Framework for Network Control. In 23rd USENIX Symposium on Networked Systems Design and Implementation (NSDI 26). USENIX

Association, Renton, WA, 1–18. https://www.usenix.org/conference/ nsdi26/presentation/chen-jiayi

[8] Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 785–794.

[9] Sharan Chetlur, Clif Woolley, Philippe Vandermersch, Jonathan Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer. 2014. cudnn: Eficient primitives for deep learning. arXiv preprint arXiv:1410.0759 (2014).

[10] Leonardo Dagum and Ramesh Menon. 1998. OpenMP: an industry standard API for shared-memory programming. IEEE computational science and engineering (1998), 46–55.

[11] Katharina Eggensperger, Philipp Müller, Neeratyoy Mallik, Matthias Feurer, René Sass, Aaron Klein, Noor Awad, Marius Lindauer, and Frank Hutter. 2021. HPOBench: A collection of reproducible multifidelity benchmark problems for HPO. arXiv preprint arXiv:2109.06716.

[12] Michael Ferdman, Almutaz Adileh, Onur Kocberber, Stavros Volos, Mohammad Alisafaee, Djordje Jevdjic, Cansu Kaynak, Adrian Daniel Popescu, Anastasia Ailamaki, and Babak Falsafi. 2012. Clearing the clouds: a study of emerging scale-out workloads on modern hardware. Acm sigplan notices 47, 4, 37–48.

[13] Matteo Frigo and Steven G. Johnson. 2005. The Design and Implementation of FFTW3. Proc. IEEE 93, 2 (2005), 216–231.

[14] Victor Giannakouris and Immanuel Trummer. 2025. �-tune: Harnessing large language models for automated database system tuning. 26 pages.

[15] F Maxwell Harper andJoseph A Konstan. 2015. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis) 5, 4 (2015), 1–19.

[16] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, et al. 2020. Array Programming with NumPy. Nature 585, 7825 (2020), 357–362. doi:10.1038/s41586-020-2649-2

[17] Albert Hartono, Boyana Norris, and Ponnuswamy Sadayappan. 2009. Annotation-based empirical performance tuning using Orio. In 2009 ieee international symposium on parallel & distributed processing. IEEE, 1–11.

[18] Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown. 2011. Sequential model-based optimization for general algorithm configuration. In International conference on learning and intelligent optimization. Springer, 507–523.

[19] Frank Hutter, Manuel López-Ibánez, Chris Fawcett, Marius Lindauer, Holger H Hoos, Kevin Leyton-Brown, and Thomas Stützle. 2014. AClib: A benchmark library for algorithm configuration. In International Conference on Learning and Intelligent Optimization. Springer, 36–40.

[20] Shintaro Iwasaki, Abdelhalim Amer, Kenjiro Taura, Sangmin Seo, and Pavan Balaji. 2019. BOLT: Optimizing OpenMP parallel regions with user-level threads. In 2019 28th International Conference on Parallel Architectures and Compilation Techniques (PACT). IEEE, 29–42.

[21] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2010), 117–128.

[22] Jef Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE transactions on big data 7, 3 (2019), 535–547.

[23] Harshad Kasture and Daniel Sanchez. 2016. Tailbench: a benchmark suite and evaluation methodology for latency-critical applications. In 2016 IEEE International Symposium on Workload Characterization (IISWC). IEEE, 1–10.

[24] Tim Kraska. 2021. Towards instance-optimized data systems. Proceedings of the VLDB Endowment 14, 12 (2021).

[25] Jiale Lao, Yibo Wang, Yufei Li, Jianping Wang, Yunjia Zhang, Zhiyuan Cheng, Wanghu Chen, Mingjie Tang, and Jianguo Wang. 2024. GP-Tuner: A Manual-Reading Database Tuning System via GPT-Guided Bayesian Optimization. Proceedings of the VLDB Endowment 17, 8 (2024), 1939–1952. doi:10.14778/3659437.3659449

[26] Quentin Lhoest, Albert Villanova Del Moral, Yacine Jernite, Abhishek Thakur, Patrick Von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, et al. 2021. Datasets: A community library for natural language processing. In Proceedings of the 2021 conference on empirical methods in natural language processing: system demonstrations. 175–184.

[27] Liam Li, Kevin Jamieson, Afshin Rostamizadeh, Ekaterina Gonina, Jonathan Ben-Tzur, Moritz Hardt, Benjamin Recht, and Ameet Talwalkar. 2020. A system for massively parallel hyperparameter tuning. Proceedings ofmachine learning and systems 2 (2020), 230–246.

[28] James Litton, Anjo Vahldiek-Oberwagner, Eslam Elnikety, Deepak Garg, Bobby Bhattacharjee, and Peter Druschel. 2016. Light-Weight Contexts: An OS Abstraction for Safety and Performance.. In OSDI. 49–64.

[29] Peter Mattson, Christine Cheng, Gregory Diamos, Cody Coleman, Paulius Micikevicius, David Patterson, Hanlin Tang, Gu-Yeon Wei, Peter Bailis, Victor Bittorf, et al. 2020. Mlperf training benchmark. Proceedings ofMachine Learning and Systems 2, 336–349.

[30] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer Sentinel Mixture Models. arXiv:1609.07843 [cs.CL]

[31] Derek G Murray, Jiri Simsa, Ana Klimovic, and Ihor Indyk. 2021. tf. data: A machine learning data processing framework. arXiv preprint arXiv:2101.12127 (2021).

[32] ROYUD Nishino and Shohei Hido Crissman Loomis. 2017. Cupy: A numpy-compatible library for nvidia gpu calculations.

[33] NVIDIA. 2021. cuBLAS. https://developer.nvidia.com/cublas

[34] NVIDIA Corporation. 2026. NVIDIA DALI: NVIDIA Data Loading Library Documentation. https://docs.nvidia.com/deeplearning/dali/ user-guide/docs/. Accessed: 2026-06-03.

[35] Heidi Pan, Benjamin Hindman, and Krste Asanović. 2010. Composing Parallel Software Eficiently with Lithe. SIGPLAN Not. 45, 6, 376–387. doi:10.1145/1809028.1806639

[36] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. 2017. Automatic diferentiation in PyTorch.

[37] PostgreSQL Global Development Group. 2026. PostgreSQL Documentation: Parallel Query. https://www.postgresql.org/docs/current/ parallel-query.html. Accessed: 2026-06-03.

[38] PyTorch Contributors. 2026. torch.utils.data: PyTorch Data Loading Utilities. https://docs.pytorch.org/docs/stable/data.html. Accessed: 2026-06-03.

[39] Mark Raasveldt and Hannes Mühleisen. 2019. Duckdb: an embeddable analytical database. In Proceedings of the 2019 international conference on management ofdata. 1981–1984.

[40] Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings ofthe 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP). 3982–3992.

[41] Matthew Rocklin et al. 2015. Dask: Parallel computation with blocked algorithms and task scheduling.. In SciPy. 126–132.

[42] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. 2012. Practical bayesian optimization of machine learning algorithms. Advances in neural information processing systems 25.

[43] Chris Thornton, Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown. 2013. Auto-WEKA: Combined selection and hyperparameter optimization of classification algorithms. In Proceedings of the 19th

ACM SIGKDD international conference on Knowledge discovery and data mining. 847–855.

[44] tpc.org. 2026. TPC-H Version 2 and Version 3. https://www.tpc.org/ tpch/. Accessed: 2026-06-08.

[45] Immanuel Trummer. 2022. DB-BERT: a Database Tuning Tool that" Reads the Manual". In Proceedings of the 2022 international conference on management ofdata. 190–203.

[46] Dana Van Aken, Andrew Pavlo, Geofrey J Gordon, and Bohan Zhang. 2017. Automatic database management system tuning through large scale machine learning. In Proceedings of the 2017 ACM international conference on management of data. 1009–1024.

[47] Pauli Virtanen, Ralf Gommers, Travis E Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, et al. 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods 17, 3 (2020), 261–272.

[48] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or fiction: Verifying scientific claims. In Proceedings ofthe 2020 conference on empirical methods in natural language processing (EMNLP). 7534– 7550.

[49] Endong Wang, Qing Zhang, Bo Shen, Guangyong Zhang, Xiaowei Lu, Qing Wu, and Yajuan Wang. 2014. Intel Math Kernel Library. Cham. doi:10.1007/978-3-319-06486-4\_7

[50] Xingfu Wu, Prasanna Balaprakash, Michael Kruse, Jaehoon Koo, Brice Videau, Paul Hovland, Valerie Taylor, Brad Geltz, Siddhartha Jana, and Mary Hall. 2025. ytopt: Autotuning scientific applications for energy eficiency at large scales. Concurrency and Computation: Practice and Experience 37, 1 (2025), e8322.

[51] Yineng Yan, William Ruys, Hochan Lee, Ian Henriksen, Arthur Peters, Sean Stephens, Bozhi You, Henrique Fingler, Martin Burtscher, Milos Gligoric, et al. 2025. VLCs: Managing Parallelism with Virtualized Libraries. In Proceedings of the 2025 ACM Symposium on Cloud Computing. 629–643.

[52] Matei Zaharia, Reynold S Xin, Patrick Wendell, Tathagata Das, Michael Armbrust, Ankur Dave, Xiangrui Meng, Josh Rosen, Shivaram Venkataraman, Michael J Franklin, et al. 2016. Apache spark: a unified engine for big data processing. Commun. ACM 59, 11 (2016), 56–65.

[53] Fan Zhang, Melissa Petersen, Leigh Johnson, James Hall, and Sid E O’Bryant. 2022. Hyperparameter tuning with high performance computing machine learning for imbalanced Alzheimer’s disease data. Applied Sciences 12, 13 (2022), 6670.

[54] Ji Zhang, Yu Liu, Ke Zhou, Guoliang Li, Zhili Xiao, Bin Cheng, Jiashu Xing, Yangtao Wang, Tianheng Cheng, Li Liu, et al. 2019. An end-to-end automatic cloud database tuning system using deep rein forcement learning. In Proceedings ofthe 2019 international conference on management ofdata. 415–432.

[55] Xianyi Zhang, Qian Wang, and Yunquan Zhang. 2012. Model-driven level 3 BLAS performance optimization on Loongson 3A processor. IEEE.

## A Benchmark Details

This appendix describes the nine LibContentionBench workloads. Input data, workload sizes, and model and algorithm settings remain fixed during tuning; only the resource controls listed below may change. The configuration-space sizes are based on the 48 CPU slots available in our experiments and may difer on other hardware.

## A.1 Workload Definitions

Joblib/BLAS Nested CPU Contention. Input and execution. The workload runs 800 independent tasks, each multiplying two fixed $3 2 0 \times 3 2 0$ matrices six times. Joblib runs the tasks in parallel, and NumPy uses the host BLAS runtime for matrix multiplication [16, 55]. Controls. The tunable controls are the Joblib worker count, thread limits for OpenMP, OpenBLAS, MKL, and BLIS, and the choice of a process or thread backend. Contention. Joblib workers and the native numerical routines they invoke share CPU cores (cpu.compute). When each outer worker also uses multiple native threads, the combined demand can exceed the avail able CPU capacity. Configuration space. The five worker and thread counts, together with two backend choices, give $2 \times 4 8 ^ { 5 } = 5 0 9 , 6 0 7 , 9 3 6$ configurations.

Dask/DuckDB Columnar Memory-Spill. Input and execution. Dask processes ten Parquet files, each containing 20,000 rows in row groups of 5,000. The data is read into Arrow tables and passed to DuckDB for a fixed aggregation [39, 41]. Controls. The tunable controls are the Dask worker count, threads per worker, worker memory limit, Arrow CPU and I/O thread counts, DuckDB memory limit, DuckDB thread count, and scratch-path layout. Contention. Dask, Arrow, and DuckDB workers share CPU cores (cpu.compute). In-memory data and spill bufers con sume host memory (mem.capacity), while spill operations using the same scratch path compete for I/O bandwidth (io.bandwidth). Configuration space. The eight controls give $4 8 ^ { 5 } \times 4 ^ { 2 } \times 2 = 8 . 1 5 \times 1 0 ^ { 9 }$ configurations.

HF/PyTorch Text Data Pipeline. Input and execution. Hugging Face Datasets processes 4,096 fixed text records with Tokenizers, and a PyTorch DataLoader supplies the results to a fixed training loop [26, 38]. Each record contains 32 input tokens, the maximum sequence length is 64, and the batch size is 32. Controls. The tunable controls are the PyTorch thread count, dataset-mapping process count, tokenizer parallelism, DataLoader worker count, prefetch factor, and use of persistent workers. Contention. Dataset mapping, tokenization, data loading, and PyTorch execution share CPU cores (cpu.compute). Worker processes and prefetched batches also consume host memory (mem.capacity) Configuration space. The six controls give $4 8 ^ { 3 } \times 4 9 \times 2 ^ { 2 } =$ $2 . 1 7 \times 1 0 ^ { 7 }$ configurations.

TorchData/S3-DDP Object-Store Pipeline. Input and execution. Two CPU training ranks each read 1,024 samples from four shards in a local S3-compatible object store. The workflow uses TorchData and fsspec for input requests and PyTorch DDP for training [36]. Controls. The tunable controls are the PyTorch thread count, S3 block size, S3 cache mode, object-listing cache policy, and DDP bucket capacity. Contention. Input decoding and training share CPU cores (cpu.compute). Both ranks access the local object store over loopback and share its transfer bandwidth (io.bandwidth). Request concurrency and prefetch depth afect the number of outstanding I/O operations (io.queue), while caches and DDP buckets consume host memory (mem.capacity). Configuration space. The five controls give $4 8 \times 1 , 0 4 4 , 4 8 1 \times 4 \times$ $2 \times 6 4 = 2 . 5 7 \times 1 0 ^ { 1 0 }$ configurations.

DALI/PyTorch GPU Decode-Inference. Input and execution. DALI decodes a dataset of 5,000 JPEG images while PyTorch and cuDNN run convolutional inference under CUDA MPS [9, 34, 36]. Each run has a ten-second warm-up followed by a 45-second measurement interval. Controls. The tunable controls are the DALI batch size, prefetch depth, decoder load, decoder thread count, DALI MPS share, inference batch size, and PyTorch MPS share. Contention. Decoding and inference kernels compete for GPU compute capacity (gpu.compute), and their bufers share device memory (gpu.memory). DALI’s prefetch depth controls the number of queued batches (io.queue). Configuration space. The six integer-valued controls give $5 1 2 ^ { 2 } \times 4 8 ^ { 2 } \times { 1 0 0 } ^ { 2 } = 6 . 0 4 \times 1 0 ^ { 1 2 }$ discrete combinations. Decoder load is an additional continuous control.

Forest-Cover Model Selection. Input and execution. Dask-ML GridSearchCV evaluates six fixed model configurations using three-fold cross-validation on data from a stratified, 120,000-row UCI Covertype subset. It then refits the selected model and predicts labels for 24,000 held-out rows [5, 8, 41]. The timed region includes all 18 cross-validation fits, refitting, and prediction. Controls. The tunable controls are the cross-validation worker count, XGBoost native-thread count, and Dask local scheduler choice (threads or processes). Contention. Concurrent cross-validation fits and their native XGBoost workers share CPU cores (cpu.compute). Each fit also retains training data and intermediate state, consuming host memory (mem.capacity). Configuration space. Each integer control ranges from 0 to 48, and the scheduler has two choices, giving $4 9 ^ { 2 } \times 2 = 4 { , } 8 0 2$ configurations.

Movie Recommendation Training. Input and execution. Implicit trains an alternating least squares (ALS) recommendation model on all 20,000,263 MovieLens20M ratings, covering 138,493 users and 26,744 movies [15]. It uses 128 factors and 15 training iterations, then generates ten recommendations for every user. Controls. The tunable controls are the ALS worker count, OpenBLAS thread count, OpenMP binding policy, and placement by CPU core or hardware thread. Contention. ALS workers and BLAS kernels share CPU capacity (cpu.compute). Binding and placement settings afect how workers use cores and NUMA domains (cpu.placement). Configuration space. The four controls give $4 9 \times 4 8 \times 3 \times 2 = 1 4 , 1 1 2$ configurations.

<table><tr><td>Benchmark</td><td>Libraries and datasets</td><td>Shared resources</td><td>Configuration space</td><td>Best performance gain</td></tr><tr><td>Joblib/BLAS Nested CPU Contention</td><td>Joblib, NumPy, OpenBLAS [16, 55]</td><td>cpu.compute</td><td> $5 . 0 9 \times 1 0 ^ { 8 }$ </td><td>16.7×</td></tr><tr><td>Dask/DuckDB Columnar Memory-Spill</td><td>Dask, PyArrow, DuckDB [39, 41]</td><td>cpu.compute,mem.capacity, io.bandwidth</td><td> $8 . 1 5 \times 1 0 ^ { 9 }$ </td><td>1.66×</td></tr><tr><td>HF/PyTorch Text Data Pipeline</td><td>HF Datasets, HF Tokenizers, PyTorch DataLoader [26, 38]</td><td>cpu.compute,mem.capacity</td><td> $2 . 1 7 \times 1 0 ^ { 7 }$ </td><td>2.35×</td></tr><tr><td>Pipeline</td><td>TorchData/S3-DDP Object-Store TorchData, fsspec/s3fs, MinIO, PyTorch cpu. compute, mem. capacity,</td><td></td><td> $2 . 5 7 \times 1 0 ^ { 1 0 }$ </td><td>1.68×</td></tr><tr><td>DALI/PyTorch GPU</td><td>DDP [36] DALI/nvJPEG, PyTorch/cuDNN</td><td>io.bandwidth, io.queue gpu.compute,gpu.memory, io.queue</td><td> $6 . 0 4 \times 1 0 ^ { 1 2 }$ </td><td>1.84×</td></tr><tr><td>Decode-Inference Forest-Cover Model Selection</td><td>Dask-ML, XGBoost, UCI</td><td>cpu.compute,mem.capacity</td><td>4,802</td><td>1.84×</td></tr><tr><td>Movie Recommendation</td><td>Covertype [5, 8, 41] Implicit, OpenBLAS, OpenMP,</td><td>cpu.compute,cpu.placement</td><td>14,112</td><td>1.32×</td></tr><tr><td>Training Scientific Claim Retrieval</td><td>MovieLens20M [10, 15, 55] SentenceTransformers, PyTorch, FAISS, cpu.compute, mem.capacity</td><td></td><td> $5 . 0 4 \times 1 0 ^ { 9 }$ </td><td>2.16×</td></tr><tr><td></td><td>SciFact [22, 36, 40, 48]</td><td></td><td></td><td></td></tr><tr><td>SIFT Visual Codebook Construction</td><td>Spark MLlib, OpenBLAS, SIFT1M [21, 52, 55]</td><td>cpu.compute,mem.capacity</td><td>2,401</td><td>1.27×</td></tr></table>

Table 4: LibContentionBench workloads, libraries and datasets, shared resources expressed in LPCL, configurationspace sizes, and performance gains over the baseline.

Scientific Claim Retrieval. Input and execution. SentenceTransformers encodes 5,183 SciFact abstracts and 300 queries with a fixed MiniLM model. FAISS then performs exact search and returns ten documents per query [22, 40, 48]. Timing includes worker-pool startup, corpus and query encoding, retrieval, and shutdown, but excludes loading the model in the parent process. Controls. The tunable controls are the encoder process count, PyTorch thread count, tokenizer thread count, batch size, and chunk size. Contention. Encoder processes and their native PyTorch and tokenizer threads share CPU cores (cpu.compute). Each process also holds its own model state and intermediate bufers, consuming host memory (mem.capacity). Configuration space. The five controls give ${ 8 \times 4 9 ^ { 2 } \times 1 2 8 \times 2 , 0 4 9 } = 5 . 0 4 \times { 1 0 } ^ { 9 }$ configurations.

SIFT Visual Codebook Construction. Input and execution. Spark MLlib builds a K-means codebook with 1,024 centers from 100,000 SIFT1M learning descriptors, each with 128 dimensions [21, 52]. Each run starts a new local JVM, loads the input, trains and saves the model, assigns every descriptor to a cluster, writes the results, and shuts down. Controls. The tunable controls are the CPU slots reserved per Spark task and the process-wide OpenBLAS thread limit. Contention. Spark tasks and OpenBLAS kernels share CPU cores (cpu.compute). Concurrent Spark tasks also compete for space in the JVM heap (mem.capacity). Configuration space. The two controls give $4 9 ^ { 2 } = 2 { , } 4 0 1$ configurations.