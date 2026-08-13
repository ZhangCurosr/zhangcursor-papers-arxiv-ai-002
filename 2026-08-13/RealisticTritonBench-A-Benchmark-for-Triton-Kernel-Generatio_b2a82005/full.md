# RealisticTritonBench: A Benchmark for Triton-Kernel Generation in Real-World AI Frameworks

Jinjun Huang   
huanghuanghuang@zju.edu.cn   
College of Computer Science and   
Technology and the State Key   
Laboratory of Blockchain and Data   
Security, Zhejiang University   
Hangzhou, China Meng Yan mengy@cqu.edu.cn   
The School of Big Data and Software   
Engineering, Chongqing University Chongqing, China Zhongzhen Wen   
wenzhongzhen@smail.nju.edu.cn   
State Key Lab for Novel Software   
Technology, Nanjing University Nanjing, China   
Xin Xia   
xin.xia@acm.org   
College of Computer Science and   
Technology and the State Key   
Laboratory of Blockchain and Data   
Security, Zhejiang University   
Hangzhou, China   
Tongtong Xu   
xutongtong9@huawei.com   
Software Engineering Application   
Technology Laboratory, Huawei   
Hangzhou, China   
Zhongxin Liu<sup>∗</sup>   
liu\_zx@zju.edu.cn   
College of Computer Science and   
Technology and the State Key   
Laboratory of Blockchain and Data   
Security, Zhejiang University   
Hangzhou, China

## Abstract

In modern AI frameworks, the GPU kernel is a key determinant of overall system performance. By combining usability, portability, and near-handwritten CUDA performance, Triton has been widely adopted for implementing GPU kernels. Recent advances have shown the potential of using large language models (LLMs) to automatically generate Triton kernels, helping reduce the manual efort required by expert kernel developers. To evaluate the quality of LLM-generated Triton kernels, several benchmarks have been proposed. However, existing benchmarks primarily target isolated kernel generation tasks and sufer from three key limitations: (1) they restrict the task to only PyTorch-to-Triton translation, thus failing to reflect the diversity and complexity of real-world Triton tasks; (2) they evaluate only the performance of individual kernels, limiting the evaluation of real-world performance of generated kernels in AI frameworks, where end-to-end performance is the core criterion for real deployment; (3) they rely on manually written evaluation scripts for single kernel, which may introduce vulnerabilities and allow models to exploit evaluation flaws to bypass correctness checks and obtain inflated scores. To address these limitations, we introduce RealisticTritonBench, the first benchmark that derives Triton kernel generation tasks from real-world pull requests in popular AI frameworks, enabling evaluation under realistic, production-like settings. RealisticTritonBench systematically extracts real-world PRs with Triton kernel modified from popular open-source AI frameworks and transforms them into kernel generation tasks with concrete engineering contexts. Each task takes a natural language requirement as input and requires to generate a

corresponding Triton kernel implementation. RealisticTritonBench also provides a complete and reproducible evaluation environment for each task. In contrast to prior benchmarks that focus on isolated kernel performance, RealisticTritonBench integrates generated kernels into their original frameworks and evaluates them under end-to-end test, enabling a more faithful assessment. We conduct a systematic evaluation of the state-of-the-art LLMs on RealisticTritonBench, revealing that they still struggle to handle the challenges in real-world Triton generation tasks.

## CCS Concepts

• Software and its engineering → Automatic programming.

## Keywords

Large Language Models, Triton, Deep Learning Kernels

ACM Reference Format:

Jinjun Huang, Zhongzhen Wen, Tongtong Xu, Meng Yan, Xin Xia, and Zhongxin Liu. 2026. RealisticTritonBench: A Benchmark for Triton-Kernel Generation in Real-World AI Frameworks. In Proceedings ofthe 41st IEEE/ACM International Conference on Automated Software Engineering (ASE ’26), October 12–16, 2026, Munich, Germany. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3832783.3837457

## 1 INTRODUCTION

Currently, modern AI models are typically deployed on training and inference frameworks, such as Pytorch [29], and Deepspeed [32]. Within these frameworks, GPU kernels constitute a critical component that determines system latency and throughput, motivating engineers to invest significant efort in developing high-performance kernels, as ineficient implementations can increase latency, reduce throughput, and raise the overall cost of large-scale model deployment [12]. However, the implementation of correct and highly optimized GPU kernels using low-level native languages such as CUDA is inherently challenging and time-consuming [38]. To address these challenges, Triton [38] has emerged as a widely adopted high-level GPU programming language. It has been prevalently

```python
# An Example for Vector Addition
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
BLOCK_SIZE: tl.constexpr):
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
output = x + y
tl.store(output_ptr + offsets, output, mask=mask)
```

## Figure 1: A Triton kernel for Vector Addition

integrated into modern AI frameworks and optimization pipelines, serving as a preferred choice for implementing high-performance GPU kernels [13, 18, 51]. As a Python-based domain-specific language (DSL) for GPU kernel development, Triton significantly simplifies the implementation of complex kernels while maintaining performance competitive with native CUDA. As shown in Figure 1, a Triton kernel typically resembles a Python function decorated with @triton.jit, where developers explicitly define block-level parallelism, memory access patterns, and tensor computation through Triton primitives. Owing to its ease of use, flexibility, and strong performance, Triton has gained substantial traction in modern AI systems and has become a key building block for optimized critical kernels in contemporary AI frameworks. However, such DSLs have not fully eliminated the complexities of performance tuning. Developers are still required to reason about low-level execution details, including memory access patterns, parallelization strategies, and hardware-specific optimizations. Hence, current research in automated Triton generation has attracted increasing attention.

More recently, a line of work has explored leveraging large language models (LLMs) to automatically synthesize Triton kernels. Several benchmarks have been proposed to evaluate models’ capabilities of generating Triton kernels[20, 28, 41, 46]. These benchmarks provide an initial testbed for assessing the functional correct ness and performance of generated kernels, helping LLM-assisted kernel generation rapidly progress. Representative approaches, including AutoTriton[21], QiMeng-Kernel [52], and KernelFalcon [30], investigate LLM-based GPU kernel generation and reveal the poten tial of LLMs for automating Triton kernel generation. However, we find these key limitations in existing benchmarks: (1) Limited task diversity. Existing benchmarks predominantly focus on translating PyTorch reference implementations into Triton kernels [20, 28, 46]. In contrast, the real-world Triton generation task encompasses a broader range of activities, including performance optimization, bug fixing, and feature extension, as these tasks typically involve modifying existing Triton kernel implementations to meet new functional or performance requirements. These practical development scenarios remain insuficiently explored in existing studies. (2) Lack of framework-level evaluation. Existing benchmarks often evaluate kernels in isolation, only considering kernel-level metrics such as functional correctness and speedup compared to PyTorch implementations[20, 28]. Such kernel-level evaluation overlooks the real impact of deploying generated kernels within AI frameworks. In practice, Triton kernels must interact with complex runtime systems, memory management strategies, distributed execution pipelines, and framework-specific abstractions. Without end-toend evaluation in real-world AI frameworks, it is dificult to assess their genuine efectiveness and robustness with kernel-level evaluation. (3) Vulnerability in evaluation pipeline. Most benchmarks rely on manually written evaluation scripts[20, 28], which may introduce vulnerabilities and allow models to exploit evaluation flaws to obtain misleading results. Some researchers have observed that models may exploit flaws in evaluation scripts to bypass correctness checks, thereby obtaining inflated scores[24, 33].

To address these limitations, we present RealisticTritonBench to enable evaluation under realistic environments. RealisticTriton-Bench has several characteristics: 1) Diverse tasks: we construct RealisticTritonBench from real-world Triton-related pull requests, covering a wide range of development activities, including performance optimization, modification, and new kernel addition, rather than limiting to PyTorch-to-Triton translation tasks. 2) Multidimensional evaluation: To better approximate real-world deployment scenarios, we provide kernel-level unit tests, model accuracy tests, and end-to-end speed tests for tasks whenever possible. By incorporating end-to-end accuracy and performance evaluation, our benchmark directly measures the real impact of generated kernels within complete frameworks. 3) Mitigate Hacking in Evaluation. Since end-to-end metrics directly reflect the model’s actual performance, improvements at this level correspond to actual performance gains. As a result, it becomes dificult to exploit evaluation flaws without genuine improvements in system performance, thereby reducing the risk of misleading results.

We evaluate a diverse set of state-of-the-art LLMs on RealisticTritonBench, such as Qwen3.5 and GPT-5.4. To systematically measure task completion, we design these metrics for comprehensive evaluation: full test pass rate (FTP), unit test pass rate (UTP), numerical robustness for the model (NR), and end-to-end acceleration speedup. Together, these metrics form a realistic, system-level evaluation framework for generated kernels. To ensure comprehensive evaluation, our end-to-end testing pipeline is applied across a diverse set of models, covering diferent architectures and capabilities, summarized in our online appendix[14]. Based on these metrics, we systematically study the performance, robustness, and failure reasons of LLMs on RealisticTritonBench. Our experiments show that models achieve only 18.71% average task success rate, maintain model accuracy in 47.65% of cases, and present limited end-to-end speedup (approximately 1× on average), which indicates that generating correct Triton kernels in real-world tasks remains challenging, and the generated implementations often lead to degraded model accuracy or increased end-to-end latency.

In summary, this paper makes the following contributions:

• We curate the RealisticTritonBench dataset, providing a foundation for evaluating the performance of AI-generated kernels in realistic environments.

• We propose a comprehensive evaluation pipeline for Triton kernel generation, which spans unit testing to end-to-end systemlevel testing, closely simulating real-world kernel validation procedures.

• We conduct an extensive evaluation of state-of-the-art language models and LLM-based agents on RealisticTritonBench, revealing that even the SOTA LLMs still struggle with real-world Triton kernel generation tasks, and the generated kernels frequently lead to degraded model accuracy and increased end-to-end latency.

## 2 BACKGROUND

## 2.1 LLM Training and Inference Frameworks

Recently LLMs have been widely applied to a variety of tasks including code generation[3], mathematical reasoning[35], video generation[42], and multimodal understanding[19]. As model scale and capability continue to grow, eficiently training and deploying these large-scale models has become increasingly significant. To address this challenge, researchers and engineers have developed various training and inference frameworks. For example, in the training task, frameworks such as PyTorch[29], Tensorflow[1] DeepSpeed[32], and Megatron-LM[37] provide essential capabilities including automatic diferentiation, gradient optimization, fault tolerance, and large-scale distributed training. Although general purpose deep learning frameworks such as PyTorch and TensorFlow are widely used for LLM inference, they are primarily designed to support a broad range of hardware platforms and model architectures. As a result, they do not specifically optimize the core decoding process of LLM inference, which may lead to suboptimal performance and ineficient resource utilization.

To improve inference eficiency, a number of specialized LLM inference engines have recently been developed[9, 18, 50, 51]. These systems introduce a variety of optimizations tailored for LLM inference and serving, including techniques such as dynamic batching[2] and inference-specific operators[48]. By incorporating these optimizations, modern inference engines provide features such as eficient batching, optimized attention computation, and distributed inference, enabling LLMs to be deployed and executed more eficiently on GPU hardware.

## 2.2 Triton for High-Performance GPU Kernel Development

As deep learning models continue to grow in scale, GPU computational eficiency has become a key factor afecting both training and inference performance. In GPU programming, a GPU kernel refers to a function that runs on the GPU and is executed in parallel by a large number of threads, typically performing data-parallel computations on tensors or matrices. These kernels constitute the fundamental building blocks of many operations in deep learning systems. Although general deep learning frameworks such as PyTorch provide a rich set of built-in operators, many performancecritical computations in real-world systems still rely on customized GPU kernels to achieve higher eficiency. For example, in scenarios like attention computation, matrix operations, and sparse computation, kernels optimized for specific data layouts and memory access patterns can significantly reduce latency.

Traditionally, high-performance GPU kernels are implemented using CUDA, Nvidia’s parallel computing platform[41]. However, writing eficient CUDA kernels is complex and requires deep exper tise in GPU architecture, including thread scheduling and memory management. To reduce this burden while retaining high performance, Triton[38] has emerged as a domain-specific language for

GPU programming. It provides a Python-based abstraction that enables developers to write high-level kernels, while the compiler handles low-level optimizations such as parallelization and memory access. By combining high-level programmability with performance close to hand-written CUDA kernels, Triton significantly reduces the development complexity of high-performance GPU operators. As a result, Triton has been increasingly adopted in modern deep learning systems to implement GPU kernels. For example, systems such as vLLM[18] and libraries such as FlashAttention[4] use Triton to implement high-performance GPU kernels in their core components. However, implementing eficient Triton kernels still requires substantial efort for developers, which motivates eforts to explore the use of large language models for automatic kernel generation.

## 3 RealisticTritonBench

RealisticTritonBench aims to evaluate the ability of large language models (LLMs) to generate Triton kernels in realistic development scenarios. The overview of the benchmark is illustrated in Figure 2. The benchmark consists of 31 real Triton kernel tasks collected from popular open-source AI frameworks. Each task corresponds to a merged pull request (PR) related to Triton kernel.

For each task, RealisticTritonBench provides three components as input: a task description, relevant context, and the definition of the target function. These components are combined to form the prompt that is provided to the LLM. LLM is required to implement the target function strictly following the specified requirements.

After generation, the produced kernels replace the original implementation in the prepared testing environment. We then execute a standardized evaluation pipeline that includes kernel-level unit tests, model-level accuracy tests, and end-to-end test. Specifically, the unit tests are first executed to verify the functional correctness of the generated kernels. The modified kernels are then integrated into the framework to evaluate model accuracy and end-to-end latency. The evaluation pipeline ensures that generated kernels satisfy the expected behavior of the task, while also capturing their impact on model accuracy and system-level performance under realistic deployment settings.

In this section, we present the task formulation and the construction pipeline of RealisticTritonBench in detail.

## 3.1 Task Formulation

3.1.1 Input Definition and Target Output. To evaluate the ability of models to generate Triton kernels in real-world development environment, we construct the task inputs to closely resemble the information available to developers during kernel implementation. Specifically, each task consists of three components: task description, context, and target function specification.

The task description provides a detailed specification of the kernel functionality, describing the intended behavior and implementation constraints of the operator. The context contains relevant information extracted from the original repository, presented in the form of file-function and file–class references. These references indicate functions and classes that may be useful for implementing the target kernel, enabling the model to access the necessary information required for development.

![](images/ad869dc789436a68689fff883e59046b60a91b87b750b12f4757ee6fa5278cfc.jpg)  
Figure 2: Task formulation and evaluation pipeline of Real isticTritonBench

The definition of the target function defines the Triton kernel interface that the model is expected to implement. Given the task description and the relevant context, the model must generate a Triton kernel that conforms exactly to the provided definition.

3.1.2 Evaluation Suite. Compared with previous benchmarks that focus only on evaluation at the single-operator level, we summarize practical development experience from real-world PRs and design three types of tests to improve the robustness of the bench mark: Unit Test, Model Accuracy Test, and Latency Test.

Unit Test is used to verify the functional correctness of kernels. Model Accuracy Test evaluates whether replacing the operator implementation leads to noticeable degradation in model performance on common tasks. Latency Test measures the end-to-end system latency after the substitution.

## 3.2 Benchmark Construction

We focus on constructing high-quality, real-world Triton kernel generation tasks and build RealisticTritonBench through a fourphase pipeline. As illustrated in Figure 3, we first collect relevant pull requests from selected open-source AI framework repositories (Phase 1). We then filter and analyze these PRs to extract concrete Triton kernel generation tasks (Phase 2). Next, we construct executable evaluation environments for each instance(Phase 3). Finally, we validate and refine the instances based on execution results to ensure the correctness and stability of the benchmark (Phase 4).

3.2.1 PR Collection. The goal of this phase is to collect pull requests (PRs) related to Triton kernel modifications from opensource repositories, in order to construct a candidate task pool grounded in real-world engineering practice. We first select several popular and actively maintained AI framework repositories as data sources, including PyTorch, vLLM, and SGLang. These repositories extensively use Triton to implement high-performance GPU kernels for both training and inference pipelines, making them representative of real-world Triton development scenarios. We therefore collect historical pull requests (PRs) from these repositories as candidate instances for further analysis.

To eficiently identify Triton-related modifications among a large number of PRs, we design an automated filtering strategy as an initial filter. Specifically, we first perform a coarse-grained filtering step based on keywords appearing in PR titles, descriptions, and commit messages, such as triton, kernel, operation, and optimization. This step helps quickly locate PRs that are potentially related to Triton. Next, we further analyze the code difs in each PR and retain only those that actually modify or introduce Triton kernel implementations, such as files containing @triton.jit kernel definitions. By combining keyword-based filtering with codelevel change analysis, our approach reduces manual checking efort while maintaining high relevance to Triton kernel modifications.

Through this process, we obtain a set of PRs closely related to Triton kernel modifications as candidate instances. These PRs cover a variety of real development scenarios, including kernel optimizations, bug fixes, and the implementation of new features, providing a diverse foundation for constructing the benchmark.

3.2.2 Kernel Task Extraction. Based on the PRs related to Triton kernel modifications collected in the previous stage, we conduct a detailed analysis of each PR to identify the objectives of the kernel changes, and subsequently transform them into well-defined Triton kernel generation tasks.

Initial Filtering. After the initial keyword-based filtering in the previous step, we obtained approximately 2,000 PRs that require manual analysis. We then manually examine each PR and further filter them according to the following evaluation criteria.

(1) Triton Kernel Relevance. The PR must contain modifications related to Triton kernels.

(2) Test Availability. The PR must include corresponding tests (e.g. at least unit tests) that can be used to verify correctness.

(3) Clear Kernel Objective. The modification introduces a clear functional or performance-related objective that can be formulated as a standalone kernel task.

Task Extraction. To extract tasks from PRs, we first conduct a manual analysis of each PR by carefully examining the PR description and the corresponding code changes. Through this process, we aim to understand the background, objectives, and motivations behind the Triton kernel modifications. For example, some PRs introduce new Triton kernels to replace existing PyTorch implementations in order to improve performance, while others may extend the functionality of existing kernels. This analysis allows us to clearly identify the target functions involved and the expected functional changes introduced by the modification.

Based on the content of the PR, we then employ an LLM to automatically generate the corresponding kernel task description. The generated description is typically summarizing the target function to be implemented or modified and the expected behavior. In this way, real-world code modifications are abstracted into executable Triton kernel generation tasks. Then, the generated descriptions are manually reviewed and revised when necessary to ensure that they accurately reflect the original intention of the PR and provide a clear and complete specification of the task. Specifically, each task description is checked for intent faithfulness to the PR goal, requirement completeness regarding behavior, constraints, and success criteria, and the absence of implementation leakage from the gold patch. Two Triton-experienced developers inspect the PR description, dif, target function, and tests, and score each criterion on a 0/1/2 scale. If any criterion averages below 1.5, they revise the task description to consensus. In addition to the task description, we also manually analyze and extract the contextual information required for implementing the task, including related functions and classes , the definition of the target function to be implemented, and the corresponding tests. We categorize the tests into three types: Unit Test, Model Accuracy Test, and Latency Test. The Unit Test relies on existing pytest-based testing commands provided by the repository. For Model Accuracy Test and Latency Test, we analyze the models associated with each PR and construct testing commands based on the repository’s existing evaluation scripts. These tests are used to verify model accuracy after kernel replacement and to measure the end-to-end system latency respectively.

![](images/1f1a6a176904435439cb589ca04dc99ee85943ccf7e387646a3e70d57b38888a.jpg)  
Figure 3: The pipeline of constructing our benchmark

3.2.3 Environment Construction. Following SWE-bench[17], we provide a runnable Docker environment for each instance. Inspired by the multi-layered environment construction approach used in NoCodeBench[5], we adopt a two-level construction strategy to improve eficiency.

We first build a shared base Docker image for instances originating from the same repository. The base image includes the required system dependencies and standardized runtime configurations, such as Python and CUDA Toolkit versions. With this image as the base image, we further construct instance-specific environments that contain the exact code version and dependencies required for running the corresponding tests.

More specifically, we initially attempt to build the environment using the repository-specific build commands. However, due to issues such as ambiguous dependency specifications or incompatible library updates, some instances fail to build. For example, newer versions of certain libraries may introduce APIs that are incompatible with previously supported configurations or older hardware, requiring specific files to be switched to compatible versions.

To address these instance-specific issues, we record the full build logs during the environment setup process and manually resolve installation errors when they occur. The corresponding fixes are then documented as executable shell commands, such as specify ing dependency versions or patching configuration files. These commands are integrated into the instance-specific build scripts to ensure that the environment can be reproduced automatically.

This stage guarantees that each instance is equipped with an independent and fully functional containerized environment, providing a reliable execution foundation for subsequent testing and evaluation.

3.2.4 Refine on Execution Feedback. After collecting candidate instances and constructing executable environments, we further

Table 1: Distribution of task categories in RealisticTriton-Bench

<table><tr><td></td><td>Optimization</td><td>Modification</td><td>New Kernel</td></tr><tr><td>Percentage</td><td>41.93%</td><td>22.58%</td><td>35.48%</td></tr></table>

Table 2: Studied Large Language Models
<table><tr><td>Model</td><td>Type</td><td>Size</td><td>Time</td></tr><tr><td>Deepseek-V3.2(non-reasoning)</td><td>non-reasoning</td><td>685B</td><td>2025-12-01</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>reasoning</td><td>397B</td><td>2026-03-09</td></tr><tr><td>Deepseek-V3.2(reasoning)</td><td>reasoning</td><td>685B</td><td>2025-12-01</td></tr><tr><td>GPT-5.4</td><td>reasoning</td><td>unpublished</td><td>2026-03-05</td></tr><tr><td>Gemini-3.1 Pro Preview</td><td>reasoning</td><td>unpublished</td><td>2026-02-19</td></tr></table>

verify the validity of the tests associated with each instance and perform additional filtering and refinement. Specifically, we first execute all Unit Tests provided for each instance and remove those that fail to pass all Unit Tests, excluding failures caused by hardwarerelated issues. Since Unit Tests reflect the expected behavior of the modified Triton kernels, instances that cannot pass these tests are considered invalid as they fail to correctly reproduce the behavior of the original PR, and are therefore excluded from the final dataset.

Next, we run the Model Accuracy Test and Latency Test and adjust the corresponding execution commands when necessary based on the observed results. This step is required because the repositories continue to evolve over time, and the interfaces or arguments of testing scripts may change accordingly. As a result, some original test commands may no longer execute correctly in the reconstructed environment. We therefore refine the testing commands according to the actual execution results to ensure that the evaluation scripts remain valid and can reliably measure model accuracy and system latency.

Through these phases, we ultimately obtained 31 task instances, which are grouped into three categories, as summarized in Table 1.

## 4 EXPERIMENT

In this section, we evaluate the performance of the state-of-the-art (SOTA) LLMs in diferent settings and analyze the results. Specifi cally, we aim to address the following three research questions.

• RQ1: How do state-of-the-art LLM-based agents perform on RealisticTritonBench?

• RQ2: How do LLMs’ performance vary across diferent task categories?

• RQ3: What are the main reasons for the failures of LLMs on RealisticTritonBench?

Table 3: Performance comparison across diferent LLMs on RealisticTritonBench.
<table><tr><td>Model</td><td>Success (%)</td><td>Applied (%)</td><td>FTP (%)</td><td>UTP (%)</td><td>NR(%)</td><td>STTFT</td><td>STPOT</td><td>Cost</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>Deepseek-V3.2 (non-reasoning)</td><td>12.90%</td><td>93.55%</td><td>45.16%</td><td>56.55%</td><td>45.45%</td><td>0.9708</td><td>0.9587</td><td>17.54M</td></tr><tr><td>Deepseek-V3.2 (reasoning)</td><td>19.35%</td><td>96.77%</td><td>41.94%</td><td>66.84%</td><td>40.00%</td><td>0.9903</td><td>0.9881</td><td>25.89M</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>25.81%</td><td>96.77%</td><td>45.16%</td><td>58.12%</td><td>58.33%</td><td>1.0110</td><td>0.9959</td><td>11.75M</td></tr><tr><td colspan="9">Closed-source Models</td></tr><tr><td>GPT-5.4</td><td>16.13%</td><td>100.0%</td><td>35.48%</td><td>52.16%</td><td>44.44%</td><td>1.375</td><td>0.8948</td><td>9.23M</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>19.35%</td><td>100.0%</td><td>48.39%</td><td>67.98%</td><td>50.00%</td><td>0.9424</td><td>0.9687</td><td>10.47M</td></tr><tr><td>Average</td><td>18.71%</td><td>97.42%</td><td>43.23%</td><td>60.33%</td><td>47.65%</td><td>1.0579</td><td>0.9612</td><td>14.98M</td></tr></table>

## 4.1 Methodology

4.1.1 Model Selection. The tasks in RealisticTritonBench require models to generate complete Triton kernel implementations based on task descriptions and repository context. The process demands strong capabilities in task understanding, Triton kernel generation, and long-context reasoning. Therefore, we select a set of state-ofthe-art LLMs for evaluation, as shown in Table 2. Among them, three models are open-source models, including DeepSeek-V3.2 (non-reasoning)[23], DeepSeek-V3.2 (reasoning)[23] and Qwen3.5- 397B-A17B[31]. The other two are closed-source models, including Gemini-3.1 Pro Preview[10] and GPT-5.4[27]. These models represent some of the most advanced large language models currently available, with strong performance on code generation and reasoning tasks. They difer in model architecture, training paradigms, and functional capabilities, which enables us to investigate how SOTA models with diferent architectures perform on Triton kernel generation tasks. For models with reasoning capabilities, we set the reasoning efort to high when configurable and otherwise simply enable thinking.

4.1.2 Scafold Selection. Finishing Triton kernel generation tasks requires models to retrieve relevant code context from the repository as essential dependency information for implementation. Due to the lack of accurate context, directly prompting an LLM often fails to produce correct and complete implementations. Therefore, we select mini-SWE-agent [47] as the evaluation scafold.

mini-SWE-agent is a lightweight implementation derived from the SWE-agent system developed by the Princeton and Stanford teams, designed to simplify the coding-agent pipeline while maintaining strong performance. On the SWE-Bench (Verified) leaderboard, it achieves over 76% pass rate. Although several agents rank higher, including live-swe-agent[45], Sonar Foundation Agent, TRAE[8], Atlassian Rovo Dev, and EPAM AI/Run Developer Agent, they are not suitable for our evaluation for the following reasons.

First, Sonar Foundation Agent, Atlassian Rovo Dev, and EPAM AI/Run Developer Agent are closed-source systems, making it impossible to obtain their implementations for experimentation. Sec ond, TRAE is inactively maintained and currently does not support models without tool-calling capabilities. Finally, although live-sweagent is SOTA and open-source, it modifies the original scafold through tool evolution. Since our goal is to evaluate the intrinsic capability of models, we follow SWE-Bench and adopt a bash-only tool interface for the scafold. Taking into account of these factors, mini-SWE-agent is ultimately selected as the scafold used in our evaluation.

4.1.3 Evaluation Metrics. Diferent from kernel-oriented benchmarks [20, 28, 46], which mainly focus on kernel-level evaluation, we assess the practical performance of generated Triton kernels within the overall framework. We introduce four evaluation metrics to assess the correctness, model numerical stability, and systemlevel performance of the generated kernels:

• Full Test Pass Rate (FTP%): The percentage of tasks whose generated implementations pass all unit tests.

• Unit Test Pass Rate (UTP%): The percentage ofunit tests passed out of the total number of unit tests for a task.

• Numerical Robustness for Model (NR, T/F): Evaluates whether the kernel modification introduces numerical instability at the model level. After integrating the generated kernel into the framework, the model is evaluated on a common benchmark. Following the criteria derived from real-world open-source PRs, if the task performance does not degrade, the metric is marked as T; otherwise, it is marked as F.

• End-to-End Acceleration Speedup: Measures the improvement in end-to-end latency after deploying the modified kernel in the AI framework compared with the reference implementation under the same workload. We report the speedup of both Time to First Token (TTFT) and Time per Output Token (TPOT), defined as

$$
S _ { \mathrm { T T F T } } = \frac { \mathrm { T T F T } _ { \mathrm { b a s e } } } { \mathrm { T T F T } _ { \mathrm { n e w } } } , \quad S _ { \mathrm { T P O T } } = \frac { \mathrm { T P O T } _ { \mathrm { b a s e } } } { \mathrm { T P O T } _ { \mathrm { n e w } } } .
$$

A value greater than 1 indicates a latency reduction compared with the baseline. (We report the average NR and speedup over tasks that pass all unit tests)

In addition, following prior work [5, 17, 49], we also report the success rate of kernel task(Success%) and token cost(Cost) for each model, the success rate of patch application(Applied%) as aggregate evaluation metrics. A task is considered successful only if all of the following conditions are satisfied:

• The Unit Test Pass Rate (UTP) matches that of the gold patch implementation.

• The Numerical Robustness check passes (�� = �).

• The end-to-end latency satisfies $S _ { \mathrm { T T F T } } \geq 0 . 9 8$ and $S _ { \mathrm { T P O T } } \geq 0 . 9 8 .$

The first condition ensures functional correctness, the second verifies numerical stability at the model level, and the third ensures that the generated kernel does not significantly degrade systemlevel inference performance. We adopt a tolerance threshold of

0.98 for latency speedup to account for minor runtime fluctuations, and discuss its sensitivity in our online appendix [14]. All experiments are conducted on a server equipped with 8× NVIDIA RTX 3090 GPUs and averaged over three runs, the average variation is 1.08% for $S _ { \mathrm { T T F T } }$ and 0.98% for $S _ { \mathrm { T P O T } }$ , indicating stable speedup measurements.

Notably, we adopt the reference implementation from the gold patch as the baseline for two reasons. First, these PRs from widely used frameworks such as vLLM underwent multiple rounds of review by experienced kernel experts, making them a credible reference grounded in real-world development practice. Second, it is the only available reference implementation in the target framework at that time, making it a meaningful baseline for assessing whether generated patches reach the accepted real-world solution.

## 4.2 RQ1: Performance on RealisticTritonBench

We evaluate five representative LLMs on RealisticTritonBench, and the results are presented in Table 3. Our findings show that under stricter and more realistic multi-dimensional evaluation criteria, even state-of-the-art LLMs exhibit limited performance on real-world Triton kernel tasks. Specifically, Qwen3.5-397B-A17B achieves the best performance, with a task success rate of 25.81%, indicating that only a small portion of tasks can be fully solved under realistic constraints.

In terms of unit test performance, the average Unit Test Pass Rate (UTP) across all models reaches 60.33%, and the Full Test Pass Rate (FTP) is 43.23%. However, the overall task success rate is only 18.71%, which is significantly lower than the FTP. This notable gap indicates that although models can often generate implementations that pass all unit tests, these implementations may still degrade model accuracy or increase end-to-end latency when deployed in real systems. In other words, passing unit tests alone is insuficient to guarantee the practical usability of generated kernels. Regarding numerical robustness, only 47.65% of the cases maintain model accuracy without degradation after replacing the original code. This further highlights that correct implementations may also introduce subtle numerical issues that negatively afect model performance. For end-to-end latency, most models achieve an average S<sub>TTFT</sub> of around 1.0579, indicating no clear performance improvement compared to the baseline. Although GPT-5.4 achieves a higher S<sub>TTFT</sub> of 1.375, this is mainly due to the limited number of correct cases included in its latency evaluation, making the result less representative. Similarly, the average $\mathbf { S } _ { \mathrm { T P O T } }$ of 0.9612 further suggests that replacing kernels often results in comparable or slightly degraded performance. This observation indicates that performance optimization at the kernel level is still challenging for current LLMs, especially when considering system-level interactions.

Overall, these results demonstrate that RealisticTritonBench effectively exposes the gap between unit test correctness and real system performance, preventing models from concentrating on unit tests alone. By incorporating system-level accuracy and latency evaluation, our benchmark provides a more comprehensive, realistic, and challenging testbed for Triton kernel generation, and better reflects the requirements of real-world deployment scenarios.

Summary: The state-of-the-art LLMs perform poorly on RealisticTritonBench, struggling to generate Triton kernels that simultaneously pass all unit tests, preserve model accuracy, and improve end-to-end latency. This highlights the dificulty of achieving system-level correctness and eficiency, and demonstrates that RealisticTritonBench efectively exposes the gap between unit test success and real-world deployment requirements.

## 4.3 RQ2: Category-wise Performance Analysis

Prior work mainly focuses on translating PyTorch kernels into Triton kernels, which corresponds to the New-kernel category in our work. Beyond this setting, we further introduce two additional task categories: Optimization and Modification. The Optimization category targets performance improvements of existing Triton kernels, while the Modification category involves fixing bugs or extending functionality (e.g., adding support for new data types) in existing kernels. In this section, we analyze the experimental results from a category-wise perspective. Table 4 presents the performance of diferent models across the three task types: Optimization, Modification, and New-kernel.

Optimization. Models achieve average 23.08% success rates on Optimization tasks, with relatively high UTP and FTP across all models. This indicates that LLMs are generally capable of generating optimized Triton kernels that preserve the original functionality based on existing implementations. However, the end-to-end speedup remains limited, with most $\bf { S } _ { T T F T }$ and S<sub>TPOT</sub> values close to 1.0, suggesting that models struggle to deliver meaningful performance improvements. Furthermore, numerical robustness remains a challenge, with only 56.19% of the generated kernels maintaining model accuracy without noticeable degradation. This indicates that even when functional correctness is preserved, the generated implementations may still introduce subtle numerical inconsistencies that afect downstream model behavior. These results highlight the dificulty of considering low-level performance optimization, numerical stability, and system-level eficiency simultaneously.

Modification. For Modification tasks, models achieve performance comparable to Optimization tasks in terms of UTP and FTP. Both task types share a common characteristic in that they require generating kernels based on existing Triton implementations, while Optimization focuses on improving performance; Modification aims to alter the original functionality, such as adding new features or fixing bugs. The comparable performance results on UTP and FTP further prove that, given existing Triton kernels as references, LLMs are generally capable of generating functionally equivalent implementations. However, numerical robustness remains a significant challenge in this setting. On average, only 20.00% of the generated kernels maintain model accuracy without degradation, with several models achieving 0% NR. This indicates that Modification tasks are more likely to introduce subtle numerical issues that degrade model accuracy. It also suggests that while LLMs can follow functional requirements, they lack a deep understanding of numerical stability and edge-case behaviors when modifying existing kernels.

New-kernel. The average task success rate of only 5.455% in Table 4 indicates that New-kernel is the most challenging category for current models. Compared with the other two task types, this setting requires the model to implement a functionally equivalent Triton kernel without any prior Triton implementation as reference. Across the 11 New-kernel tasks, on average, only 20% of the generated kernels are able to pass all unit tests, which is the lowest among all categories. Moreover, even among the limited set of correct kernels, the end-to-end latency metrics are generally below 1.0, and the average numerical robustness (NR) is only 40%, suggesting that newly generated kernels often fail to match or surpass the performance of existing implementations. These results highlight the fundamental dificulty for LLMs in generating high-quality Triton kernels from scratch when no Triton-specific reference implementation is available.

```python
# Target Function
@triton.jit
def kernel_unified_attention_2d(...):
cur_batch_in_all_start_index =
tl.load(query_start_len_ptr + seq_idx)
cur_batch_in_all_stop_index =
tl.load(query_start_len_ptr + seq_idx + 1)
cur_batch_query_len = cur_batch_in_all_stop_index -
cur_batch_in_all_start_index
seq_len = tl.load(seq_lens_ptr + seq_idx)
q_end_pos = tl.min(q_end_pos, cur_batch_query_len)
```  
Figure 4: An example of incorrect usage of the Triton API.

Summary: LLMs perform well in functional correctness when modifying existing Triton kernels but struggle with performance optimization and numerical robustness. When no Triton-based implementation is provided, performance drops significantly, revealing limited capability in generating high-quality kernels from scratch.

## 4.4 RQ3: Failure Analysis

To better understand the limitations of LLMs in Triton kernel generation, we manually inspect the failed cases and identify the key reasons that lead to their poor performance on RealisticTriton-Bench. Through this analysis, we reveal the limitations of current LLMs in real-world Triton kernel generation tasks.

Insuficient fundamental capability in Triton programming. On average, 56.77% of the cases fail to pass all unit tests, indicating that generating fully correct Triton kernels remains highly challeng ing for current LLMs. A fundamental prerequisite for completing these tasks is the ability to produce compilable and executable code that strictly adheres to Triton’s programming constraints, including its type system, supported control flow, and memory access semantics. However, our analysis of failure cases shows that a substantial portion of errors originate from violations of these low-level constraints. Among the cases that fail to pass all unit tests, 64.71% fall into this category. Concretely, models frequently hallucinate non-existent APIs (e.g. tl.info, tl.finfo), misuse existing primi tives (e.g. incorrect usage of tl.min), or produce code that violates Triton’s strict typing rules. In addition, several cases exhibit incorrect memory access patterns, such as mismatched tensor shapes or invalid pointer arithmetic in tl.load, which result in runtime errors or silent incorrect behavior. There are also instances of unsupported control flow constructs (e.g. break) and missing essential components (e.g. decorators), further preventing successful execution. These results indicate that current LLMs lack a systematic and reliable understanding of Triton’s low-level programming model. In particular, they struggle to internalize the strict constraints imposed by the compiler and runtime, including type safety, valid API usage, and explicit memory layout requirements.

For example, as illustrated in Figure 4, the model incorrectly uses the tl.min API. In kernel\_unified\_attention\_2d, both cur\_batch\_query\_len and q\_end\_pos are scalar values (i.e., 0-D Triton tensors). However, tl.min in Triton is designed as a reduction API, expecting an input tensor together with a reduction axis, rather than two scalar operands for binary comparison. Therefore, tl.min(q\_end\_pos, cur\_batch\_query\_len) does not match the intended semantics of taking the minimum of two scalar values. The correct API here should be tl.minimum, which performs an elementwise minimum between two operands.

Lack of understanding of kernel semantics in real-world repository code. In addition to violations of low-level programming constraints, we observe a 29.41% of failures that stem from incorrect implementations of kernel semantics in concrete code repositories. Successfully completing these tasks requires the model to comprehensively reason about kernel behavior in repositories, including boundary conditions, corner cases, and fine-grained implementation details that are critical to correctness and numerical behavior. We observe that LLMs frequently make mistakes in implementation details. As illustrated in Figure 5, the generated implementation fails to handle segment processing. Instead of returning early for invalid segments, it incorrectly writes back incorrect M\_seg, L\_seg, and acc\_seg, thereby corrupting the subsequent segment-level reduction. This case further highlights that LLMs lack a comprehensive understanding of kernel semantics in real-world implementations. In real-world kernel generation tasks, correct implementations require careful consideration of both the task objectives and the surrounding code context to properly handle diferent scenarios, including boundary conditions and edge cases. LLMs frequently struggle to correctly handle corner cases and often ignore implicit conditions, ultimately leading to functional inconsistencies or incorrect numerical results. This indicates that LLMs lack a comprehensive understanding of kernel semantics in real-world repository contexts.

Lack of attention to performance and numerical stability during implementation. In our experiments, models successfully generate Triton kernels that pass all correctness checks in 43.23% of cases. However, in 25.16% of cases, replacing the original implementation with the generated kernel leads to degraded model accuracy or end-to-end performance. Through case analysis, we find that generated kernels often introduce unnecessary operations or miss critical optimizations compared to reference kernels.

For example, as shown in Figure 6, the gold implementation encloses the penalty-related computation within an if use\_penalty branch, thereby avoiding redundant arithmetic operations and predicate computations. In contrast, the generated implementation executes these computations unconditionally, resulting in increased latency. Furthermore, we observe that generated kernels may degrade model accuracy due to subtle numerical issues introduced during implementation. For example, changes to the specialization strategy (e.g., conversion to tl.constexpr and modifying do\_not\_specialize) may afect Triton’s execution behavior, leading to small numerical deviations that accumulate and impact model accuracy. These cases show that LLMs lack of enough attention to performance and numerical stability in kernel implementation.

Table 4: Category-wise performance of diferent models on RealisticTritonBench across task types: Optimization, Modification, and New-kernel.
<table><tr><td>Task Type</td><td>Model</td><td>Success (%)</td><td>Applied (%)</td><td>FTP (%)</td><td>UTP (%)</td><td>NR(%)</td><td> ${ \bf S } _ { \mathrm { T T F T } }$ </td><td>STPOT</td></tr><tr><td colspan="9">Optimization</td></tr><tr><td></td><td>Deepseek-V3.2 (non-reasoning)</td><td>15.38%</td><td>100.0%</td><td>53.85%</td><td>75.26%</td><td>71.43%</td><td>0.9442</td><td>0.9510</td></tr><tr><td></td><td>Deepseek-V3.2 (reasoning)</td><td>7.692%</td><td>100.0%</td><td>53.85%</td><td>76.47%</td><td>42.86%</td><td>0.9891</td><td>0.9794</td></tr><tr><td></td><td>Qwen3.5-397B-A17B</td><td>38.46%</td><td>100.0%</td><td>69.23%</td><td>82.88%</td><td>66.67%</td><td>1.018</td><td>0.9920</td></tr><tr><td>GPT-5.4</td><td></td><td>23.08%</td><td>100.0%</td><td>46.15%</td><td>53.65%</td><td>50.00%</td><td>1.812</td><td>0.9001</td></tr><tr><td></td><td>Gemini-3.1-Pro-Preview</td><td>30.77%</td><td>100.0%</td><td>61.54%</td><td>75.61%</td><td>50.00%</td><td>0.9964</td><td>1.005</td></tr><tr><td></td><td>Average</td><td>23.08%</td><td>100.0%</td><td>56.92%</td><td>72.78%</td><td>56.19%</td><td>1.152</td><td>0.9654</td></tr><tr><td colspan="9">Modification</td></tr><tr><td></td><td>Deepseek-V3.2 (non-reasoning)</td><td>28.57%</td><td>100.0%</td><td>71.43%</td><td>71.43%</td><td>0.000%</td><td>1.027</td><td>0.9824</td></tr><tr><td></td><td>Deepseek-V3.2 (reasoning)</td><td>42.86%</td><td>100.0%</td><td>57.14%</td><td>79.92%</td><td>0.000%</td><td>1.002</td><td>1.008</td></tr><tr><td></td><td>Qwen3.5-397B-A17B</td><td>42.86%</td><td>100.0%</td><td>57.14%</td><td>62.34%</td><td>50.00%</td><td>0.9825</td><td>1.004</td></tr><tr><td></td><td>GPT-5.4</td><td>14.29%</td><td>100.0%</td><td>28.57%</td><td>54.10%</td><td>0.000%</td><td>1.026</td><td>1.019</td></tr><tr><td></td><td>Gemini-3.1-Pro-Preview</td><td>28.57%</td><td>100.0%</td><td>57.14%</td><td>73.19%</td><td>50.00%</td><td>1.033</td><td>1.024</td></tr><tr><td>Average</td><td></td><td>31.43%</td><td>100.0%</td><td>54.29%</td><td>68.20%</td><td>20.00%</td><td>1.014</td><td>1.007</td></tr><tr><td colspan="9">New-kernel</td></tr><tr><td></td><td>Deepseek-V3.2 (non-reasoning)</td><td>0.000%</td><td>81.81%</td><td>18.18%</td><td>24.97%</td><td>0.000%</td><td>0.9791</td><td>0.9500</td></tr><tr><td></td><td>Deepseek-V3.2 (reasoning)</td><td>18.18%</td><td>90.90%</td><td>18.18%</td><td>47.14%</td><td>100.0%</td><td>0.9825</td><td>0.9990</td></tr><tr><td></td><td>Qwen3.5-397B-A17B</td><td>0.000%</td><td>90.90%</td><td>9.091%</td><td>26.16%</td><td>0.000%</td><td>1.007</td><td>1.014</td></tr><tr><td>GPT-5.4</td><td></td><td>9.091%</td><td>100.0%</td><td>27.27%</td><td>49.17%</td><td>50.00%</td><td>0.8053</td><td>0.8559</td></tr><tr><td></td><td>Gemini-3.1-Pro-Preview</td><td>0.000%</td><td>100.0%</td><td>27.27%</td><td>55.65%</td><td>50.00%</td><td>0.7380</td><td>0.8359</td></tr><tr><td>Average</td><td></td><td>5.455%</td><td>93.94%</td><td>20.00%</td><td>40.62%</td><td>40.00%</td><td>0.9025</td><td>0.9310</td></tr></table>

# Reference Implementation   
def kernel\_unified\_attention\_3d(...):   
# Reference Implementation   
blocks\_per\_segment = cdiv\_fn(seq\_len,   
NUM\_SEGMENTS\_PER\_SEQ \* BLOCK\_SIZE)   
if segm\_idx\*blocks\_per\_segment\*BLOCK\_SIZE >= seq\_len:   
return   
# Generated implementation   
segment\_len = cdiv\_fn(num\_blocks, NUM\_SEGMENTS\_PER\_SEQ)   
start\_block = segment\_idx \* segment\_len   
end\_block = tl.minimum(start\_block + segment\_len,...)   
tl.store(segm\_max\_ptr + segm\_offset, M, ...)   
tl.store(segm\_expsum\_ptr + segm\_offset, L, ...)  
Figure 5: An example where the generated implementation incorrectly writes back invalid segments instead of skipping

Summary: LLM failures in real-world Triton kernel generation stem from three key limitations: insuficient mastery of Triton programming constraints, incomplete understanding of kernel semantics in real-world codebases, and lack of attention to performance eficiency and numerical stability.

```python
def _penalties_and_temperature_kernel(...):
# Reference Implementation
if use_penalty:
req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
output_bin_counts = tl.load(output_bin_counts_ptr +
req_state_idx * output_bin_counts_stride +
block,mask=mask)
output_bin_mask = output_bin_counts > 0
# Generated Implementation
req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
output_bin_counts = tl.load( output_bin_counts_ptr +
req_state_idx * output_bin_counts_stride + block,
mask=mask,)
```  
Figure 6: An example where the generated implementation performs redundant computation

## 5 Discussion

## 5.1 Necessity of Framework-Level Evaluation

Among the failures reported in our evaluation, a portion of them, such as missing boundary masks, could in principle be caught by stronger isolated kernel-level tests. However, the failures related to numerical deviations, whose acceptability is decided by downstream computation, can hardly be detected by kernel-level tests.

For example, in one case from our evaluation, i.e., \_fused\_moe\_ lora\_kernel (shown in Figure 7), the generated kernel introduces two numerical deviations. One arises from parallelism in computation: the generated kernel replaces the gold implementation’s sequential accumulation over the K dimension with parallel Ksplitting merged via atomic\_add. Under this parallel strategy, the commit order of the partial sums is decided by GPU scheduling

Figure 7: An example of numerical deviations whose acceptability is decided by downstream computation

Table 5: Reward-hacking strategies catalogued by SOL-ExecBench [22]
<table><tr><td>Category</td><td>Exploit Description</td></tr><tr><td>Concurrency</td><td>Offloads computation to background threads, side streams, or forked processes to escape the timed region.</td></tr><tr><td>State &amp; Caching</td><td>g Returns cached or lazily materialized results, or behaves correctly only when a check is detected.</td></tr><tr><td>Environment</td><td>Monkey-patches timing or comparison utilities, or silently downgrades numerical precision for speed.</td></tr></table>

races. Since floating-point addition is not associative, the outputs deviate slightly from the reference and even vary across runs on the same input. The other arises from rounding of partial sums: the gold implementation rounds each value to BF16 before writing it, whereas the generated kernel rounds each value directly to the output element type. This diference in rounding behavior can introduce a small but systematic numerical discrepancy when the output dtype is not BF16. Both numerical deviations are tiny per call, and the kernel passes all unit tests, yet they accumulate across dozens of stacked MoE-LoRA layers and thousands of autoregressive decoding steps, eventually degrading GSM8K accuracy. These failures can hardly be detected by the tolerance threshold of a kernel-level unit test, because it is hard to derive an appropriate threshold for two reasons. First, deriving an acceptable error threshold requires framework-level analysis. Determining how much per-call deviation can be tolerated without degrading task accuracy requires tracing how the error propagates through the actual model’s stacked layers, discrete routing decisions, and autoregressive decoding process, none of which is a property of the kernel. Second, the acceptable error threshold is deployment-specific rather than kernel-specific. Although the same kernel may be reused across models, tasks, and runtime configurations, these contexts determine how errors propagate and accumulate. Models may invoke the kernel across more layers, longer decoding sequences allow errors to accumulate over steps, and diferent serving precisions change the magnitude of each deviation. These system-level efects are dificult, if not impossible, to capture with kernel-level unit tests, making framework-level evaluation essential.

## 5.2 Reward Hacking Mitigation

Reward hacking occurs when a model exploits loopholes in the evaluation environment to maximize its score without genuinely solving the underlying task[22]. Prior work[22] systematically categorizes hacking strategies in kernel generation into three groups: concurrency-based hacks, state-and-caching hacks, and environment manipulation, as summarized in Table 5. These hacks arise from a fundamental limitation of kernel-level testing: it cannot fully capture the correctness and performance of a kernel as it operates within the complete framework. Although kernel-level metrics are often strongly correlated with the metrics that matter in deployment, they are not equivalent. To mitigate the gap, RealisticTritonBench evaluates each kernel in a setting that closely mirrors its real-world usage. Specifically, (1) the generated kernel is integrated through the same framework call paths used in deployment; (2) it is validated using the same benchmarks that framework developers use to test kernels before deployment; (3) performance is measured using the end-to-end serving latency, i.e., time-to-first-token (�<sub>TTFT</sub>) and time-per-output-token (�<sub>TPOT</sub>), and model accuracy, which actual deployment cares about.

```python
for k in range(0, grid_k):
accumulator += tl.dot(a, b)
a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
# Reference Implementation
accumulator = accumulator.to(tl.bfloat16) # fixed to bf16
tl.store(c_ptrs, accumulator, mask=c_mask)
# Generated Implementation
tl.atomic_add(c_ptrs,
accumulator.to(c_ptrs.dtype.element_ty), mask=c_mask) #
rounds to output dtype, not bf16
```

```python
# Hacked kernel: escape the timed stream
def hacked_kernel(x):
out = torch.empty_like(x)
with torch.cuda.stream(side_stream):
real_kernel[grid](x, out, ...)
return out
# Kernel-level harness (simplified)
start.record() # on default stream
hacked_kernel(x)
end.record() # on default stream
torch.cuda.synchronize()
t = start.elapsed_time(end) # launch overhead only
assert_close(hacked_kernel(x), ref) # passes
```  
Figure 8: A stream-injection hack that defeats kernel-level timing checks.

This design mitigates all three categories ofreward hacking listed in Table 5. First, concurrency-based hacks are inefective because an external client measures TTFT and TPOT as the wall-clock time from request submission to response delivery. This interval encompasses work performed by any thread, CUDA stream, or process. Consequently, hidden work in concurrency-based hacks is either included in the measured latency or results in incorrect outputs. Second, state-and-caching hacks are unlikely to succeed because kernel outputs are immediately consumed by downstream computations across thousands of invocations under evolving runtime states. Cached, stale, or lazily materialized results therefore quickly cause numerical divergence and fail the model-accuracy evaluation. Third, environment-manipulation attacks are neutralized because the timing client and accuracy harness run in separate processes beyond the reach of the generated kernel. The kernel therefore cannot improve its score by modifying the evaluation utilities. Similarly, silently reducing numerical precision for additional speed is inefective when the resulting errors propagate through the model and degrade its end-to-end accuracy.

Figure 8 illustrates a stream-injection attack. The hacked wrapper launches the actual computation on a separate CUDA stream and returns immediately. An in-process timer that synchronizes only the default stream consequently measures little more than the launch overhead, while a later correctness check may still pass after the delayed computation has completed. This strategy is inefective in RealisticTritonBench because the external client continues measuring time until it receives the final response. Thus, computation hidden on a separate stream is either fully reflected in TTFT and TPOT or causes incorrect outputs that fail the accuracy evaluation.

## 6 Threats To Validity

The first threat arises from the automatic generation of task descriptions. In our construction pipeline, the task description of each instance is generated by an LLM from the corresponding pull request (PR) information and code changes. The LLM may misinterpret the developer’s intent, or omit implicit implementation details. To mitigate this, we manually review all generated task descriptions and refine them when necessary to ensure they accurately reflect the intent of the original PR. The second threat concerns the generalization of our results across models. We evaluate only a set of representative LLMs, so the conclusions may depend on the selected models and may not fully generalize to all existing models. To alleviate this, we select mainstream models from diferent model families with varying capabilities to improve the representativeness of the evaluation. The third threat stems from the evaluation scafold. Main experiments are conducted with mini-SWE-agent, while diferent scafolds may adopt diferent retrieval strategies and execution workflows that could influence model performance. We choose mini-SWE-agent because it represents the state-of-theart level among open-source coding agents and is also the oficial agent adopted by SWE-bench for evaluating model performance. Furthermore, we additionally evaluate GPT-5.4 with its native scaf fold Codex to examine the influence of the scafold choice and provide the results in our online appendix [14]. The last threat is potential data contamination. Our benchmark is constructed from publicly available repositories, so the evaluated models may have been exposed to the relevant code during training. We analyze this risk and provide results in our online appendix [14], which suggest that contamination is unlikely to afect our conclusions.

## 7 RELATED WORK

## 7.1 Triton Kernel Generation Datasets

To evaluate the quality of Triton kernels generated by LLMs, several benchmarks have been proposed. KernelBench [28], the first benchmark for GPU kernel generation, collects 250 AI workloads of three complexity levels and requires models to generate optimized kernels from PyTorch reference implementations. TritonBench [20] targets Triton specifically, with tasks curated from highly starred Triton repositories and selected PyTorch implementations. More recently, FlashInferBench [46] extends the pipeline to integration within practical AI frameworks, enabling end-to-end evaluation.

Despite these eforts, the first two benchmarks formulate tasks as PyTorch-to-Triton translation and evaluate kernels in isolation with kernel-level metrics. These kernel-level benchmarks do not capture real-world development scenarios such as performance optimization. FlashInferBench integrates generated kernels into practical frameworks, but still difers from our work in two aspects. (1) Task formulation. It targets generating new kernels for a set of predefined kernel definitions, whereas RealisticTritonBench derives tasks from merged PRs in popular AI frameworks, covering Optimization, Modification, and New Kernel. (2) Evaluation metrics. It mainly relies on the kernel-level fast\_p metric complemented with request-level latency, whereas RealisticTritonBench evaluates unit-test correctness, model numerical robustness, and end-to-end TTFT/TPOT latency, directly measuring the impact on accuracy and serving performance after deployment.

## 7.2 LLM for Automated Triton Generation

LLMs have been widely applied to general code generation tasks [15, 16, 25, 26, 34, 39, 40, 44] and have also shown promising potential for Triton kernel generation. Existing approaches can be categorized into three groups: Domain Model Training, Agent-Based Pipelines, and Agentic RL Training.

Domain Model Training fine-tunes LLMs on systematically collected or synthesized Triton-specific data. KernelLLM [7] performs instruction tuning on compiler-aligned PyTorch-Triton pairs, AutoTriton [21] combines supervised fine-tuning with GRPO-based reinforcement learning [36], and TritonRL [43] further introduces hierarchical reward decomposition. Agent-Based Pipelines employ LLMs as coding agents that iteratively refine kernels through a generate-execute-refine loop driven by tools and runtime feedback, as exemplified by KernelFalcon [30], TritorX [11], and AKG Kernel Agent [6]. Agentic RL further optimizes the agent’s decisionmaking via reinforcement learning, e.g., QiMeng-Kernel [52] adopts a two-stage paradigm that first learns hardware-aware optimization strategies and then implements them into eficient kernels.

## 8 CONCLUSION

In this paper, we introduce RealisticTritonBench, a benchmark for evaluating LLM-based Triton kernel generation under realistic development settings. Unlike prior benchmarks that focus mainly on PyTorch-to-Triton translation and kernel-level evaluation, Realistic-TritonBench is constructed from real pull requests collected from open-source AI frameworks and covers diverse development scenarios, including kernel optimization, modification, and new operator implementation. We further design a comprehensive evaluation pipeline that integrates unit testing with end-to-end model accuracy and system-level latency evaluation. Through experiments on several representative LLMs, we provide a systematic analysis of their capabilities and limitations in practical Triton kernel development tasks. Our results show that current LLMs lack the capability to generate correct and performance-eficient Triton kernels in real-world tasks. We envision RealisticTritonBench to facilitate the development of Triton kernel generation methods that are efective under realistic scenarios.

## 9 DATA AVAILABILITY

Our code and data are available at https://doi.org/10.5281/zenodo. 19221469 and https://github.com/ZJU-CTAG/RealisticTritonBench

## Acknowledgments

This research is supported by the National Natural Science Foundation of China (No.92582107) and Zhejiang Provincial Natural Science Foundation of China (No.LZ25F020003).

## References

[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jefrey Dean, Matthieu Devin, Sanjay Ghemawat, Geofrey Irving, Michael Isard, et al. 2016. {TensorFlow}: a system for {Large-Scale} machine learning. In 12th USENIX symposium on operating systems design and implementation (OSDI 16). 265–283.

[2] Ahsan Ali, Riccardo Pinciroli, Feng Yan, and Evgenia Smirni. 2020. Batch: Machine learning inference serving on serverless platforms with adaptive batching. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 1–15.

[3] Ruisheng Cao, Mouxiang Chen, Jiawei Chen, Zeyu Cui, Yunlong Feng, Binyuan Hui, Yuheng Jing, Kaixin Li, Mingze Li, Junyang Lin, et al. 2026. Qwen3-Coder Next Technical Report. arXiv preprint arXiv:2603.00729 (2026).

[4] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashat tention: Fast and memory-eficient exact attention with io-awareness. Advances in neural information processing systems 35 (2022), 16344–16359.

[5] Le Deng, Zhonghao Jiang, Jialun Cao, Michael Pradel, and Zhongxin Liu. 2025. Nocode-bench: A benchmark for evaluating natural language-driven feature addition. arXiv preprint arXiv:2507.18130 (2025).

[6] Jinye Du, Quan Yuan, Zuyao Zhang, Yanzhi Yi, Jiahui Hu, Wangyi Chen, Yiyang Zhu, Qishui Zheng, Wenxiang Zou, Xiangyu Chang, et al. 2025. AKG kerne Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis. arXiv preprint arXiv:2512.23424 (2025).

[7] Zacharias V. Fisches, Sahan Paliskara, Simon Guo, Alex Zhang, Joe Spisak, Chris Cummins, Hugh Leather, Gabriel Synnaeve, Joe Isaacson, Aram Markosyan, and Mark Saroufim. 2025. KernelLLM: Making Kernel Development More Accessible. https://huggingface.co/facebook/KernelLLM

[8] Pengfei Gao, Zhao Tian, Xiangxin Meng, Xinchen Wang, Ruida Hu, Yuanan Xiao, Yizhou Liu, Zhao Zhang, Junjie Chen, Cuiyun Gao, et al. 2025. Trae agent: An llm-based agent for software engineering with test-time scaling. arXiv preprint arXiv:2507.23370 (2025).

[9] Georgi Gerganov and contributors. 2023. llama.cpp: LLM Inference in C/C++. https://github.com/ggml-org/llama.cpp. GitHub repository.

[10] Google. 2026. Gemini 3.1 Pro Preview. https://ai.google.dev/gemini-api/docs models/gemini-3.1-pro-preview. accessed: 2026-03.

[11] Alec Hammond, Aram Markosyan, Aman Dontula, Simon Mahns, Zacharias Fisches, Dmitrii Pedchenko, Keyur Muzumdar, Natacha Supper, Site Cao, Haishan Zhu, et al. 2026. Agentic operator generation for ml asics. Proceedings ofMachine Learning and Systems 8 (2026), 1583–1594.

[12] Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, and Yanning Chen. 2024. Liger kernel: Eficient triton kernels for llm training. arXiv preprint arXiv:2410.10989 (2024).

[13] Jiawei Hu, Hong Jia, Mahbub Hassan, Lina Yao, Brano Kusy, and Wen Hu. 2025. LightLLM: A versatile large language model for predictive light sensing. In Proceedings ofthe 23rd ACM Conference on Embedded Networked Sensor Systems. 158–171.

[14] Jinjun Huang, Zhongzhen Wen, Tongtong Xu, Meng Yan, Xin Xia, and Zhongxin Liu. 2026. Online Appendix for “RealisticTritonBench: A Benchmark for Triton-Kernel Generation in Real-World AI Frameworks”. https://anonymous.4open. science/r/RealisticTritonBench-2583/appendix/appendix.md

[15] Zhonghao Jiang, David Lo, and Zhongxin Liu. 2025. Agentic Software Issue Resolution with Large Language Models: A Survey. arXiv preprint arXiv:2512.22256 (2025).

[16] Zhonghao Jiang, Xiaoxue Ren, Meng Yan, Wei Jiang, Yong Li, and Zhongxin Liu. 2025. Issue Localization via LLM-Driven Iterative Code Graph Searching. In 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 3034–3045.

[17] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. Swe-bench: Can language models resolve real-world github issues?. In International Conference on Learning Representations, Vol. 2024. 54107–54157.

[18] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Eficient memory management for large language model serving with pagedattention. In Proceedings ofthe 29th symposium on operating systems principles. 611–626.

[19] Hongyu Li, Jinyu Chen, Ziyu Wei, Shaofei Huang, Tianrui Hui, Jialin Gao, Xi aoming Wei, and Si Liu. 2025. Llava-st: A multimodal large language mode for fine-grained spatial-temporal understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8592–8603.

[20] Jianling Li, Shangzhan Li, Zhenye Gao, Qi Shi, Yuxuan Li, Zefan Wang, Jiacheng Huang, WangHaojie WangHaojie, Jianrong Wang, Xu Han, et al. 2025. Tritonbench: Benchmarking large language model capabilities for generating triton operators. In Findings of the Association for Computational Linguistics: ACL 2025. 23053–23066.

[21] Shangzhan Li, Zefan Wang, Ye He, Yuxuan Li, Qi Shi, Jianling Li, Yonggang Hu, Wanxiang Che, Xu Han, Zhiyuan Liu, et al. 2025. Autotriton: Automatic triton programming with reinforcement learning in llms. arXiv preprint arXiv:2507.05687 (2025).

[22] Edward Lin, Sahil Modi, Siva Kumar Sastry Hari, Qijing Huang, Zhifan Ye, Nestor Qin, Fengzhe Zhou, Yuan Zhang, Jingquan Wang, Sana Damani, et al. 2026. SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits. arXiv preprint arXiv:2603.19173 (2026)

[23] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Cheng gang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).

[24] Wei Liu, Jiawei Xu, Yingru Li, Longtao Zheng, Tianjian Li, Qian Liu, and Junxian He. 2026. Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations. arXiv preprint arXiv:2602.05885 (2026).

[25] Yingwei Ma, Rongyu Cao, Yongchang Cao, Yue Zhang,Jue Chen, Yibo Liu, Yuchen Liu, Binhua Li, Fei Huang, and Yongbin Li. 2025. Swe-gpt: A process-centric language model for automated software improvement. Proceedings ofthe ACM on Software Engineering 2, ISSTA (2025), 2362–2383.

[26] Yingwei Ma, Qingping Yang, Rongyu Cao, Binhua Li, Fei Huang, and Yongbin Li. 2025. Alibaba lingmaagent: Improving automated issue resolution via comprehensive repository exploration. In Proceedings ofthe 33rd ACM International Conference on the Foundations ofSoftware Engineering. 238–249.

[27] OpenAI. 2026. GPT-5.4 Model. https://developers.openai.com/api/docs/models/ gpt-5.4. Accessed: 2026-03.

[28] Anne Ouyang, Simon Guo, Simran Arora, Alex L Zhang, William Hu, Christopher Re, and Azalia Mirhoseini. 2025. KernelBench: Can LLMs Write Eficient GPU Kernels?. In Forty-second International Conference on Machine Learning.

[29] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems 32 (2019).

[30] PyTorch Team and Contributors. 2025. KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents. https://github.com/meta-pytorch/kernelagent. Accessed: 2025-07.

[31] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. https://qwen. ai/blog?id=qwen3.5

[32] Jef Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 3505–3506.

[33] Reddit Community. 2025. Sakana discovered its AI CUDA engineer cheating. https://www.reddit.com/r/OpenAI/comments/1iwc24f/sakana\_discovered\_ its\_ai\_cuda\_engineer\_cheating/

[34] Haifeng Ruan, Yuntong Zhang, and Abhik Roychoudhury. 2025. Specrover: Code intent extraction via llms. In 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE). IEEE, 963–974.

[35] Zhihong Shao, Yuxiang Luo, Chengda Lu, ZZ Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. 2025. Deepseekmath-v2: Towards self verifiable mathematical reasoning. arXiv preprint arXiv:2511.22570 (2025).

[36] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024).

[37] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053 (2019).

[38] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. 2019. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages. 10–19.

[39] Junyi Wang, Jialun Cao, and Zhongxin Liu. 2026. iCoRe: An Iterative Correlation Aware Retriever for Bug Reproduction Test Generation. Proceedings ofthe ACM on Software Engineering 3, FSE (2026), 4231–4252.

[40] Xinchen Wang, Pengfei Gao, Xiangxin Meng, Chao Peng, Ruida Hu, Yun Lin, and Cuiyun Gao. 2025. Aegis: An agent-based framework for bug reproduction from issue descriptions. In Proceedings of the 33rd ACM International Conference on the Foundations ofSoftware Engineering. 331–342.

[41] Zhongzhen Wen, Yinghui Zhang, Zhong Li, Zhongxin Liu, Linna Xie, and Tian Zhang. 2025. MultiKernelBench: A Multi-Platform Benchmark for Kernel Gener ation.

[42] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. 2025. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328 (2025).

[43] Jiin Woo, Shaowei Zhu, Allen Nie, Zhen Jia, Yida Wang, and Youngsuk Park. 2025. Tritonrl: Training llms to think and code triton without cheating. arXiv preprint arXiv:2510.17891 (2025).

[44] Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2025. Demystifying llm-based software engineering agents. Proceedings of the ACM on Software Engineering 2, FSE (2025), 801–824.

[45] Chunqiu Steven Xia, Zhe Wang, Yan Yang, Yuxiang Wei, and Lingming Zhang. 2025. Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly? arXiv preprint arXiv:2511.13646 (2025).

[46] Shanli Xing, Yiyan Zhai, Alexander Jiang, Yixin Dong, Yong Wu, Zihao Ye, Charlie F Ruan, Yingyi Huang, Yineng Zhang, Liangsheng Yin, et al. 2026. Flashinferbench: Building the virtuous cycle for ai-driven llm systems. Proceedings of Machine Learning and Systems 8 (2026), 2016–2064.

[47] John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. 2024. Swe-agent: Agent-computer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

[48] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, et al. 2025. Flashinfer: Eficient and customizable attention engine for llm inference serving. Proceedings ofMachine Learning and Systems 7 (2025).

[49] Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Shulin Xin, Linhao Zhang, Qi Liu, Aoyan Li, Lu Chen, Xiaojian Zhong, et al. [n. d.]. Multi-SWEbench: A Multilingual Benchmark for Issue Resolving. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

[50] Li Zhang, Youhe Jiang, Guoliang He, Xin Chen, Han Lv, Qian Yao, Fangcheng Fu, and Kai Chen. 2025. Eficient Mixed-Precision Large Language Model Inference with TurboMind. arXiv preprint arXiv:2508.15601 (2025).

[51] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jef Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. 2024. Sglang: Eficient execution of structured language model programs. Advances in neural information processing systems 37 (2024), 62557–62583.

[52] Xinguo Zhu, Shaohui Peng, Jiaming Guo, Yunji Chen, Qi Guo, Yuanbo Wen, Hang Qin, Ruizhi Chen, Qirui Zhou, Ke Gao, et al. 2026. Qimeng-kernel: Macro thinking micro-coding paradigm for llm-based high-performance gpu kernel generation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 29168–29176.