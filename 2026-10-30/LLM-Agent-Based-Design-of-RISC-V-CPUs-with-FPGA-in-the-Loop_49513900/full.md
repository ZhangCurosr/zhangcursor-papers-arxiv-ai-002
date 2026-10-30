# LLM Agent-Based Design of RISC-V CPUs with FPGA-in-the-Loop Evaluation

Leo Itoh

The University of Tokyo

Tokyo, Japan

itoh@mtl.t.u-tokyo.ac.jp

Junichiro Kadomoto

The University of Tokyo

Tokyo, Japan

kadomoto@mtl.t.u-tokyo.ac.jp

Abstract—Long-run RTL simulation of CPU candidates can take hours, while short diagnostic runs may miss performance degradation that emerges later in execution. This evaluation cost can delay design decisions in LLM-agent-based CPU development. We present a development environment in which LLM agents implement CPU candidates on an FPGA and use onboard performance measurements to guide design decisions. We applied it to the human-supervised development of a single-cycle RV32I core into a three-wide out-of-order RV32IM core. The final design reaches 5.61 CoreMark/MHz at 130 MHz on a Artix-7 FPGA. For the same CPU candidate running 8,000 CoreMark iterations, FPGA evaluation, including implementation, was 9.6× faster than single-threaded RTL simulation. In one case, board measurements revealed long-run performance degradation in a candidate that improved in a short diagnostic run. The result guided candidate selection and led to RTL-based root-cause analysis and a fix.

Index Terms—LLM, RISC-V, FPGA

## I. INTRODUCTION

The performance and implementation cost of a CPU core depend on the choice of pipeline, issue width, branch prediction, and memory system. The open RISC-V instruction set architecture has made it practical to design cores that explore these choices. Kadomoto et al. designed RISC-V cores with different pipeline structures and compared their performance, power, and area [1]. Fu et al. evaluated single-issue, multiissue, and out-of-order cores with a common SoC and implementation flow [2]. Beyond such comparisons, BOOM-Explorer searches BOOM microarchitecture parameters by multi-objective Bayesian optimization with active learning and a surrogate model [3]. ChatDSE explores a parameterized design space using LLM-based estimates of performance and power [4].

LLMs also generate and modify RTL. ChatCPU supports CPU design and verification from a specification and reported a RISC-V CPU tape-out [5]. El-Helwani and Khalid compared prompt structures for generating a single-cycle RV32I core and evaluated its FPGA implementation [6]. BluesFL maps instruction execution to code structure to support LLM-based fault localization in CPUs [7].

However, full-length RTL simulation can take hours to evaluate a CPU candidate. Short diagnostics reduce evaluation time but may miss performance degradation caused by predictor and scheduler state that develops during longer runs.

Prior work has also integrated FPGA implementation into agent-based CPU development. In auto-arch-tournament, agents modify CPU RTL and perform functional checks, followed by FPGA place and route [8]. Its standard evaluation code runs CoreMark on Verilator and multiplies iterations per cycle by the Fmax from place and route. Programming and running a physical board are not part of this evaluation path.

FPGA execution also supports cycle-exact system simulation in FireSim [9] and DNN accelerator design space exploration in Quickloop [10]. Lyra uses instruction streams from a generative model for differential testing and coverage collection on RISC-V FPGA cores [11]. Our focus is on feeding long-run board measurements back to LLM agents for CPU candidate selection and RTL analysis.

We make three contributions. First, we present a development environment that connects agent-driven RTL changes to on-board evaluation under human supervision. Second, we report the development of an RV32I core into a three-wide outof-order RV32IM core that exceeds published open-source 32- bit soft cores in CoreMark/MHz, with long-run board results guiding selection and analysis. Third, we compare the time to evaluate the same candidate by RTL simulation and on an FPGA.

## II. DEVELOPMENT FRAMEWORK

Agents extend a single-cycle RV32I core using the verification and FPGA evaluation flow in Fig. 1.

## A. Human and Agent Roles

The human set goals at each stage of CPU development and proposed directions such as multiple issue and out-oforder execution. The human also approved work and decided when to stop. Agents wrote and modified the CPU RTL and carried out detailed design, implementation, optimization, analysis, and verification. We used both single-agent and orchestrator–worker configurations during development. In the latter, GPT models [12] or Claude Opus 5.5 [13] coordinated design and review, while the Qwen3.8-Flash-Next model [14] handled implementation and analysis. Tasks and reviews were exchanged through herdr v0.9.3 [15].

The orchestrator issues tasks that specify the change hypothesis, target files, verification conditions, acceptance criteria, and compute allocation. Each worker implements changes and evaluates candidates in a separate Git worktree, then reports the results.

![](images/3460931ff200ea36a8c481f43462c436514ec84245990559ef02a242d730ab03.jpg)  
Fig. 1. Agent-based CPU development environment. Functional checks and on-board performance measurements feed back into agent decisions through a shared candidate record.

![](images/d636c0bed240f90d8875fe9aa29c803c224342b6b2f33a908987738a70a3de73.jpg)

(a)  
![](images/522c2179eec64a1756d3daf8cea02d48314a4e449136ed8a6dc36782a109b60c.jpg)  
(b)  
Fig. 2. FPGA testbed: (a) Nexys Video board and (b) host connection and on-chip design.

TABLE I  
RECORDED DEVELOPMENT ACTIVITY
<table><tr><td>Activity</td><td>Observed count</td></tr><tr><td>Human → orchestrator 11 messages</td><td></td></tr><tr><td></td><td>Orchestrator → worker 18 tasks, 24 reviews, 10 continuations or re- sumptions</td></tr><tr><td>Worker evaluations</td><td>115 RTL simulations, 40 FPGA implementa- tions, 4 board runs</td></tr><tr><td>RTL candidates</td><td>29 total; 27 simulated, 6 implemented on FPGA, 2 evaluated on board</td></tr></table>

Observed records for the three adopted changes, not a complete census. FPGA implementations include checkpoint resumptions.

## B. FPGA Evaluation

Fig. 2 shows the evaluation setup. Agents in WSL2 call Vivado, hw\_server, and the serial API on the Windows side. Resource usage and timing come from Vivado reports, and cycle counts and CRCs come from the board UART.

Functional verification uses lint, directed tests for arithmetic, memory dependence, and branch recovery, and a per-retirement comparison with a reference model. In these tests, each retired instruction’s PC and register writeback are checked against a custom RV32IM instruction-set model implemented in Python. Board measurements informed performance-based candidate selection before full-length RTL simulation completed. Final acceptance still required matching UART output from a full-length simulation and two independent board runs using the frozen RTL and firmware.

## C. Experiment Management

Each CPU design candidate has an identifier and a link to its parent design. We save the frozen RTL, firmware, implementation settings, logs, hashes, and the accept or reject decision. Tasks and reviews include the latest measurement results and failure reasons. Handover notes record measurements, running jobs, and next steps. Table I summarizes the recorded development activity for three changes incorporated into the final CPU: a load-wait predictor fix, register-move aliasing, and a two-way BTB.

TABLE II COREMARK PERFORMANCE  
![](images/a09b54ef6e611fc5df3a1fe565b51c8735a3a9e52eacaf35e81a70df18b0b712.jpg)

Fig. 3. Final CPU pipeline. Orange blocks highlight register-move aliasing and the two-way BTB.  
![](images/13cf9484741c6920e0fa3616978d052e54b3a1960eeb005faeb76f2a32a96a38.jpg)

(b) Embench (18 benchmarks)  
![](images/27d4e560960f12c524526540db32cce0c6e737ee15676ce498c277c7f1edfa52.jpg)  
Fig. 4. Evaluation time including FPGA implementation. Axes are broken and scaled separately for each workload.

## III. EXPERIMENTAL SETUP

We run CoreMark for 8,000 iterations, corresponding to a measured interval of 10.95 s and satisfying its 10 s minimum runtime [16]. We also evaluate 18 of the 19 Embench-IoT benchmarks, since cubic does not fit in the 32 KiB instruction memory. The final design targets a bare-metal RV32IM environment. It has a 32 KiB instruction memory and a 64 KiB data memory. The program and initial data are stored as FPGA memory initialization values.

CoreMark performance is reported in iterations per second (iter/s). Elapsed time is the board-reported tick count divided by the MMCM frequency. IPC is the number of instructions retired during the measured interval divided by its tick count.

<table><tr><td>Design</td><td>CoreMark/MHz MHz</td><td></td><td>LUT</td></tr><tr><td>This work (on board)</td><td></td><td></td><td></td></tr><tr><td>Single-cycle RV32I</td><td>1.36</td><td>25</td><td>1</td></tr><tr><td>1-wide, 5-stage</td><td>2.87</td><td>100</td><td>一</td></tr><tr><td>2-wide in-order</td><td>3.32</td><td>100</td><td></td></tr><tr><td>3-wide OoO</td><td>5.44</td><td>100</td><td></td></tr><tr><td>3-wide OoO (final)</td><td>5.61</td><td>130</td><td>30,344</td></tr><tr><td>Existing soft cores (published)</td><td></td><td></td><td></td></tr><tr><td>VexRiscv full max perf [17]</td><td>2.57</td><td>200</td><td>1,935</td></tr><tr><td>NaxRiscv RV32 [18]</td><td>5.02</td><td>155</td><td>≈13,300</td></tr></table>

Published soft-core results use speed grade -3 (this work uses -1) and different memory configurations, compiler settings, and design scopes.

We obtain both counts from a simulation of the same RTL and firmware.

For the controlled CoreMark comparison, we use one frozen RTL and firmware configuration of the final design. The firmware was built with RISC-V GCC 13.2.0 at -O3 with additional unrolling (-funroll-loops and related options) for 8,000 iterations. The RTL simulation model was built with Verilator 5.020 and run on a single thread with assertions disabled. All timings were measured on one host with an AMD Ryzen 7 5700X (8 cores, 16 threads) and 64 GB of memory, using Ubuntu 24.04 on WSL. Verilator and GCC ran on Ubuntu, and Vivado ran on Windows. We measured the wallclock duration of each evaluation step. Model build, full-length simulation, FPGA re-implementation, and result capture were each measured once. Programming and result capture with an existing bitstream were measured three times. FPGA evaluation includes implementation, memory initialization updates, programming, and result capture. For Embench, one FPGA implementation is shared across all 18 benchmarks, with perbenchmark memory-initialization updates instead of repeated placement and routing. RTL evaluation includes model build and full-length execution. We timed successful evaluation runs after the shared firmware build. The FPGA re-implementation reused previously established settings and met setup and hold timing on the first attempt. Embench uses CPU\_MHZ=130.

## IV. RESULTS

## A. CPU Performance

Table II reports CoreMark/MHz for representative development stages and existing soft cores. Starting from the singlecycle RV32I core, we added a five-stage RV32IM pipeline, dual issue, and out-of-order execution with register renaming. We then restructured the design to increase its implementation frequency.

The final design is a three-wide out-of-order RV32IM core with register renaming (Fig. 3). It combines timing-aware pipeline partitioning with register-move aliasing and a twoway BTB. It reaches 730.6 iter/s at 130.137 MHz over 8,000 CoreMark iterations. Its 5.61 CoreMark/MHz is the highest in Table II, 12% above the published NaxRiscv result. The design uses 10,320 FFs, 32 BRAM tiles, and 4 DSPs, with setup/hold slack of +0.003/+0.035 ns and an IPC of 1.60. On Embench-IoT 1.0 [19], it achieves a geometric-mean frequency-normalized score of 1.759 relative to the reference Cortex-M4 over 18 benchmarks.

![](images/27ef50e6e4cbb2fac3b862961dbd75f48bc2d007eb4836ee727e7ce5cce0af08.jpg)  
Fig. 5. Agent feedback in the predictor case study: (a) long-run evaluation, (b) RTL diagnosis, and (c) fix selection. Quoted text is excerpted from the records ([. . . ] marks omissions). Unquoted text summarizes the records.

![](images/a959d5476f43bc1fbe6a4e6915e4916c124445489789cb8c2d97319e16886939.jpg)  
Fig. 6. Per-iteration RTL measurements for the predictor case study. The modified core enters a persistent slower state at iteration 238.

## B. Evaluation Time

Fig. 4 compares RTL and FPGA evaluation times using identical RTL and firmware. For CoreMark, RTL evaluation took 3.91 h and FPGA evaluation took 24.5 min, a speedup of 9.6×. For Embench, RTL evaluation took 15.05 h and FPGA evaluation took 65.7 min, a speedup of 13.7×. With an existing bitstream, programming and result capture for CoreMark take a median of 27.5 s.

## C. Case Study

We evaluated a predictor-modified core with TAGE-lite and a return address stack. It reduced cycles in a short 10-iteration diagnostic but increased the cycle count by 9.98% over 8,000 iterations. The baseline is the core before this extension, an intermediate design preceding the final core in Table II. The implementation agent used two independent board runs to reject the predictor-modified core and retain the baseline. Fulllength RTL simulation reproduced the board tick count more than three hours later.

The orchestrator then assigned the mismatch to a worker for per-iteration RTL analysis (Fig. 5(a)). Fig. 6 shows the transition. The modified core used 2.62% fewer cycles over iterations 2–237, then entered a persistent slower state at iteration 238. Over iterations 239–2,000, it used 10.29% more cycles.

State tracing and ablation requested by the orchestrator identified an absorbing state in the load-wait predictor (Fig. 5(b)). A load replay writes a distance-7 prediction that makes the load wait for older stores to commit. This prevents further replays and retraining while keeping a memory reservation station occupied. Clearing the predictor entry instead of storing distance 7 eliminated the abrupt slowdown in a 400-iteration RTL run of the modified core. The baseline running the modified-core firmware remains flat through 300 iterations, ruling out the firmware difference as the cause of the transition. The orchestrator requested a non-absorbing fix. After reviewing the worker’s comparison, it selected a fix with no additional state (Fig. 5(c)).

Vivado timing results also guided RTL changes. The register-move source substitution had a setup slack of −0.291 ns at 130.137 MHz. The orchestrator requested separating the substitution data from the instruction queue. The revised version uses a separate LUTRAM and a ready bit. The final design combines this revision with the two-way BTB and passed on-board acceptance.

## V. CONCLUSION

We presented an FPGA-in-the-loop environment for humansupervised, agent-based CPU development. In the development case, on-board measurements guided candidate selection and subsequent RTL analysis and fixes. Including the cost of the initial FPGA implementation, FPGA evaluation was 9.6× faster than RTL simulation for CoreMark and 13.7× faster for the 18 Embench benchmarks. The resulting core reached 5.61 CoreMark/MHz at 130 MHz on board, above published VexRiscv and NaxRiscv results. Selected development conversations from the long-run case study and representative CPU RTL snapshots are available online.<sup>1</sup>

## ACKNOWLEDGMENT

We used OpenAI Codex and Anthropic Claude Code to assist with drafting and figure preparation, with all content reviewed and revised by the authors. This work was supported in part by JST CRONOS Grant Number JPMJCS25K2 and JST ASPIRE Grant Number JPMJAP2515.

## REFERENCES

[1] J. Kadomoto, H. Irie, and S. Sakai, “Evaluation of different microarchitectures for energy-efficient RISC-V cores,” in Proc. IEEE Int. Symp. Embedded Multicore/Many-core Systems-on-Chip (MCSoC), 2022, pp. 78–84, doi: 10.1109/MCSoC57363.2022.00022.

[2] Z. Fu, R. Tedeschi, G. Ottavi, N. Wistoff, C. Fuguet, D. Rossi, and L. Benini, “Ramping up open-source RISC-V cores: Assessing the energy efficiency of superscalar, out-of-order execution,” in Proceedings of the 22nd ACM International Conference on Computing Frontiers, ser. CF ’25. New York, NY, USA: Association for Computing Machinery, 2025, pp. 12—-20. [Online]. Available: https://doi.org/10.1145/3719276.3725186

[3] C. Bai, Q. Sun, J. Zhai, Y. Ma, B. Yu, and M. D. F. Wong, “BOOM-Explorer: RISC-V BOOM microarchitecture design space exploration,” ACM Trans. Des. Autom. Electron. Syst., vol. 29, no. 1, Dec. 2023. [Online]. Available: https://doi.org/10.1145/3630013

[4] M. Tang, W. Chen, L. Wu, L. Huang, and K. Zeng, “ChatDSE: A zero-shot microarchitecture design space explorer powered by GPT4.0,” ACM Trans. Des. Autom. Electron. Syst., vol. 30, no. 4, Jul. 2025. [Online]. Available: https://doi.org/10.1145/3735640

[5] X. Wang, G.-W. Wan, S.-Z. Wong, L. Zhang, T. Liu, Q. Tian, and J. Ye, “ChatCPU: An agile CPU design and verification platform with LLM,” in Proceedings of the 61st ACM/IEEE Design Automation Conference, ser. DAC ’24. New York, NY, USA: Association for Computing Machinery, 2024. [Online]. Available: https://doi.org/10.1145/3649329.3658493

[6] T. El-Helwani and M. A. S. Khalid, “LLM-Aided design and evaluation of a RISC-V processor,” IEEE Access, vol. 14, pp. 93 247–93 283, 2026.

[7] Z. Liu, X. Mao, D. Yang, J. He, Y. Qin, G. Zhang, Y. Lei, J. Xu, and J. Wu, “Debug Like a Human: Scaling LLM-based fault localization to processor design via block-level instruction-oriented slicing,” in Proceedings of the 63rd ACM/IEEE Design Automation Conference, ser. DAC ’26. New York, NY, USA: Association for Computing Machinery, 2026. [Online]. Available: https://doi.org/10.1145/3770743.3804145

[8] F. S. Bonetto, “HWE Bench: An Unbounded Benchmark for LLM Hardware Development on RISC-V,” https://github.com/FeSens/ auto-arch-tournament, 2026. [Online]. Available: https://hwebench.com

[9] S. Karandikar, H. Mao, D. Kim, D. Biancolin, A. Amid, D. Lee, N. Pemberton, E. Amaro, C. Schmidt, A. Chopra, Q. Huang, K. Kovacs, B. Nikolic, R. Katz, J. Bachrach, and K. Asanovic, “FireSim: FPGA-Accelerated cycle-exact scale-out system simulation in the public cloud,” in 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), 2018, pp. 29–42.

[10] K. Inayat, F. B. Muslim, T. Mahmood, and J. Chung, “FPGA-assisted design space exploration of parameterized AI accelerators: A quickloop approach,” Journal of Systems Architecture, vol. 155, p. 103260, 2024.

[11] J. Huo, Y. Gao, X. Liu, S. Wang, Y. Bao, X. Gao, and K. Shi, “Lyra: A hardware-accelerated RISC-V verification framework with generative model-based processor fuzzing,” in Proceedings of the 63rd ACM/IEEE Design Automation Conference, ser. DAC ’26. New York, NY, USA: Association for Computing Machinery, 2026. [Online]. Available: https://doi.org/10.1145/3770743.3803956

[12] OpenAI, “GPT-6.1 Sol,” Model documentation, accessed Oct. 22, 2026. [Online]. Available: https://developers.openai.com/api/docs/ models/gpt-6.1-sol

[13] Anthropic, “Introducing Claude Opus 5.5,” Sep. 2026, accessed Oct. 20, 2026. [Online]. Available: https://www.anthropic.com/claude-opus-5-5

[14] Z. Qiu, Z. Wang, X. Li, Y. Li, Y. Xu, Y. Wang, H. Zhang, R. Men, B. Mao, C. Zhang et al., “On the design of Qwen3.8- Next architecture: Evaluation, efficiency, and training stability,” Alibaba Group, Tech. Rep., Aug. 2026, arXiv:2608.30320v1. [Online]. Available: https://arxiv.org/abs/2608.30320v1

[15] herdrdev, “herdr,” Software, version 0.9.3, accessed Oct. 21, 2026. [Online]. Available: https://github.com/herdrdev/herdr/releases/tag/v0.9. 3

[16] EEMBC, “CoreMark,” Software, accessed Oct. 20, 2026. [Online]. Available: https://github.com/eembc/coremark

[17] SpinalHDL, “VexRiscv: Area usage and maximal frequency,” Project documentation, accessed Oct. 21, 2026. [Online]. Available: https: //github.com/SpinalHDL/VexRiscv#area-usage-and-maximal-frequency

[18] ——, “NaxRiscv: Performance and area,” Project documentation, accessed Oct. 21, 2026. [Online]. Available: https://spinalhdl.github.io/ NaxRiscv-Rtd/main/NaxRiscv/performance/index.html

[19] Embench, “Embench-IoT,” Software, version 1.0, commit 0466a18, accessed Oct. 21, 2026. [Online]. Available: https://github.com/embench/ embench-iot/tree/0466a18e4f6b47e19598d7c6ba72916d54b68f65