# Silinx: An Open, Browser-Based HDL Design Platform for Teaching Digital Systems on FPGA Boards

Pedro Maló

Departamento de Engenharia Electrotécnica e de Computadores (DEEC)

Faculdade de Ciências e Tecnologia (FCT)

Universidade NOVA de Lisboa, Portugal

Abstract—Many digital systems courses teach on FPGA boards whose vendor tool is no longer maintained: the Digilent Basys2 and Nexys2 (Xilinx Spartan-3E) and Nexys3 and Atlys (Spartan-6) need Xilinx ISE 14.7, discontinued since 2013 and hard to install on the computers and tablets students use today. The author presents Silinx, a free (AGPL-3.0) HDL design platform for teaching that keeps ISE’s Project Navigator workflow but runs in a web browser with nothing to install. A module can be written in VHDL or Verilog, in C or C++ through Silinx HLS, which is compatible with AMD’s Vitis HLS and passes all 73 of its introductory examples, or drawn as a schematic, an ASM chart, a state diagram, a Boolean function with its Karnaugh map, or a Petri net, from which its HDL is generated. Designs are checked with explanations written for beginners, simulated, run on an emulated board, built, and programmed onto a real board over WebUSB. The implementation flow is open and runs in the browser: Yosys and nextpnr compiled to WebAssembly, with Silinx’s own device data, timing model and bitstream writer for the Spartan-3E and Spartan-6, built by black-box observation of ISE’s documented tools; Lattice iCE40 boards use the existing open iCE40 tools, and further families are planned. ISE is the oracle: Silinx’s bitstreams are byte-identical to those ISE writes for the same placed and routed design, and pass ISE’s design rule check. Against ISE at the same optimization goal, the open flow uses 15% fewer slices with 14% shorter critical paths when optimizing for area, and 21% fewer slices with 8% shorter critical paths when optimizing for speed (geometric means over 45 open-source designs built by both flows). The author also describes how the platform was developed with AI coding agents under test-first project rules.

Index Terms—FPGA, HDL, digital systems education, open-source EDA, high-level synthesis, bitstream, Spartan-3E, Spartan-6, WebAssembly

## I. Introduction

Introductory digital systems courses teach combinational logic, state machines and their description in a hardware description language (HDL), and end with a circuit running on a real FPGA board. Many such courses use Digilent’s Basys2 and Nexys2 boards, built around the Xilinx Spartan-3E, or the Nexys3 and Atlys, built around the Spartan-6. The boards still work and teach the subject well, and replacing them for a whole class is expensive. Their software has not kept up: both families are supported only by Xilinx ISE 14.7, whose development stopped in 2013, and not by its successor, Vivado. ISE does not run on macOS or natively on Apple silicon, installs awkwardly on recent Windows and Linux, is a heavy download for a student’s laptop, and nothing in it will be fixed. Course time that should go into digital design goes into installing the tools.

Silinx is the author’s answer to this problem. It is an HDL design platform, and its toolchain is one part of it: projects of VHDL, Verilog and C/C++ sources and of drawings, editors with help for beginners, a simulator, a board emulator, implementation and board programming. It follows the ISE 14.x Project Navigator, so that course material, lab guides and teachers’ habits still apply, but it runs in a web browser, including synthesis, placement, routing, bitstream generation and programming. Its open toolchain builds for the Basys2 (XC3S100E, XC3S250E), Nexys2 (XC3S500E, XC3S1200E), Nexys3 (XC6SLX16) and Atlys (XC6SLX45), and for the Lattice iCE40 of the iCEBreaker; other boards can use a local ISE 14.7 through Silinx’s local server, and further families are planned. A student opens a web page and can go from a drawing to a working board without installing anything.

The contributions of this paper are:

• a browser-based HDL design platform for an introductory course, in which drawings (schematics, ASM charts, state diagrams, Boolean functions, Petri nets) and $\mathrm { C } / \mathrm { C } + +$ functions are the base of a module and its HDL is generated from them (Section III);

• an open implementation flow for the Spartan-3E and Spartan-6 boards, from HDL to bitstream with no Xilinx tool at any step, built on Yosys and nextpnr and on a bit database obtained by black-box observation of ISE’s documented tools (Section IV);

• a verification method that uses ISE as an oracle: byte-identical bitstreams, a clean design rule check (DRC), and the course’s examples as the regression suite (Section V);

• a quality-of-results comparison with ISE on an opensource design corpus, per optimization goal (Section VI); and

• an account of how the platform was developed with AI coding agents under explicit, test-first project rules (Section VII).

## II. Background and Related Work

Educational design tools: Logisim and its successors, Digital and CircuitVerse, are widely used to teach logic with simulated schematics; where they reach an FPGA board, they rely on the vendor’s tools to do so. Online simulators such as EDA Playground run HDL simulations in the browser but do not target a board. Icestudio [1] provides graphical design for Lattice iCE40 boards on top of the open IceStorm flow. None of these keeps the workflow of ISE, which existing Spartan-3E and Spartan-6 courses are written for, and none produces bitstreams for these families.

Open FPGA toolchains: Project IceStorm [2] documented the Lattice iCE40 bitstream, Project Trellis [3] the Lattice ECP5 and Project X-Ray [4] the Xilinx 7- series, each by building many small designs with the vendor’s tools and comparing the resulting bitstreams. With Yosys [5] for synthesis and nextpnr [6] or VPR [9] for place and route, they form complete open flows; YoWASP [8] compiles several of these tools to WebAssembly. Earlier work recovered parts of the bitstream formats of older Xilinx families to recover netlists from bitstreams [10], [11]. XDL, the text design format ISE provides for users’ own tools, was the basis of research tool kits such as Torc [12] and RapidSmith [13]. To the author’s knowledge no open flow produced Spartan-3E bitstreams before Silinx.

## III. The Learning Environment

An ISE-like project navigator, anywhere: Projects are made of VHDL and Verilog modules (mixed language), with implementation and simulation roles, a UCF constraints file, and import and export of ISE projects (.xise, including ISE schematics). The processes (Check Syntax, Synthesize, Implement Design, Generate Programming File, Configure Target Device), the design summary and the reports follow ISE, so that lab guides written for ISE still apply. The same page has two interfaces, a modern one organised around a lab session’s tasks and a classic, ISE-like one, in light and dark themes, in English and Portuguese, with touch and narrow screens supported. The same application runs as a public site (installable as an app, also ofline), as a single HTML file, or as a local Node.js server, on Apple-silicon Macs, Windows, Linux, Chromebooks and Android tablets.

Drawings are the base, HDL is generated: Firstsemester courses start from drawings and come to code later. Silinx has a schematic editor in the style of ISE’s (with ISE .sch import and export and a live, Logisim-style simulation of the drawing), an ASM chart editor with stepby-step simulation, a state diagram (Moore/Mealy) editor, a Boolean algebra editor, and a Petri net editor. The Boolean algebra editor takes a truth table or an expression per output and shows its Karnaugh map with the groups of the minimal sum of products or product of sums (Quine– McCluskey with exact cover); when a function has several minimal solutions, each can be browsed beside the map and the one chosen is used in the HDL and the schematic. The canonical forms are shown in full with their cost, and maps of three and four variables can be shown in 3D, on a cylinder and on a torus, where the adjacency that wraps around the flat map’s edges becomes visible. Each drawing always has its VHDL or Verilog module, generated from it, so that the drawing stays the single source; the Design pane shows one row per module, its drawing first (Fig. 1a). Conversely, an HDL module can be converted to a schematic, and an HDL state machine to a state diagram or an ASM chart. A student thus sees, at every step, the HDL that corresponds to what they drew.

C and C++ as sources: A C or C++ function can be a design source through Silinx HLS, which follows AMD’s Vitis HLS (which does not target these families): the generated VHDL or Verilog module has the ports and control protocol Vitis HLS would give it, and the C is the module’s base, as a drawing is. It supports the PIPELINE, UNROLL, ARRAY\_PARTITION, ARRAY\_RESHAPE, DATAFLOW, ALLOCATION and EXPRESSION\_BALANCE pragmas, hls::stream FIFOs and RTL blackboxes. The test bench’s main() runs as a C simulation, and its calls are replayed on the generated HDL as a C/RTL co-simulation. On AMD’s 73 Vitis HLS introductory examples, all 73 pass C simulation, synthesis and co-simulation. Students can thus meet high-level synthesis on the boards they already have.

Help for beginners: The common errors (a missing ; or end, an undeclared signal, std\_logic vs. std\_logic\_vector, width mismatches, a missing use clause, several drivers) keep their ISE-style message but come with an Explanation and a How to fix, in English or Portuguese, and many have quick fixes in the editor. Check Syntax and the live editor also run the design checks an instructor would otherwise point out by hand: an inferred latch, an incomplete sensitivity list, a combinational loop, an output never assigned, a clock used as data, and blocking and non-blocking assignments in the wrong kind of Verilog always block. The editor also has symbol renaming and go-to-definition.

Simulation and the Board Emulator: SSim, Silinx’s own simulator, is event-driven and four-valued, with delta cycles, VHDL and Verilog test benches, an ISim-like waveform window, stimuli drawn on the waveforms (as ISE’s HDL Bencher did), and a wizard that writes self-checking test benches. The Board Emulator runs the design on a drawing of the Basys2, Nexys2, Nexys3, Atlys, Spartan-3E Starter Kit or iCEBreaker, wired by the constraints: students click switches and buttons and watch LEDs and the multiplexed seven-segment display (Fig. 1b); a VGA monitor shows the picture a design scans out. Besides an event-driven engine, a compiled engine reads the design at word level through Yosys and runs it as one JavaScript function per clock cycle. On the course examples it runs at 59 to 141 million cycles per second (20 for an 8-bit microprocessor), measured in Node.js on an Apple M4 Pro, so counters, clock dividers and debouncers run at the boards’ 50 MHz with their real values. The netlists after synthesis, mapping and place and route can be emulated and simulated too, the routed one in SSim with its delays from an SDF file, so that a clock too fast for the routed design shows up in simulation.

![](images/4628e24a8496546627b3486432acf02394aabbe0e88d6356ec5cae8da039fa3e.jpg)  
(a) State diagram editor

![](images/af52bf010dc6c222b76015d0a09df4a0626706e1ca1677f1982c113bd49b842b.jpg)  
(b) Board Emulator  
Fig. 1. Silinx in the browser, on the trafic example (a pedestrian crossing on a Basys2). (a) The controller drawn as a state diagram, from which its HDL is generated; the Design pane lists one row per module, with badges for its drawing first and its generated HDL last. (b) The whole design on a drawing of the Basys2, wired by its constraints file (switches, buttons, LEDs, seven-segment display), on the compiled engine. Screenshots of the public site, release 15.23.0.

Programming the board from the browser: The Digilent Adept USB protocol and the JTAG programming of the FPGA and of the board’s Platform Flash are implemented in JavaScript over WebUSB, so Chrome or Edge on a computer, or Chrome on an Android tablet, programs the board directly; the iCEBreaker’s SPI flash is programmed the same way. Finally, a Model Context Protocol (MCP) server lets an AI assistant perform the same operations as the interface on a student’s or teacher’s projects.

## IV. The Open Toolchain

Fig. 2 shows the flow. Yosys [5] runs as WebAssembly (YoWASP [8]), and nextpnr [6] as WebAssembly builds of a fork of its Himbaechel architecture framework [7] in which Silinx’s Spartan-3E and Spartan-6 microarchitectures are written; the rest is JavaScript, so the same code runs in the browser and on a server. The toolchain is also published on npm as a package of its own (silinx-fpga, Apache-2.0).

## A. Front end and synthesis

Silinx’s own VHDL and Verilog parsers and elaborator, already needed by the editor and the simulator, write the elaborated design as one flat Verilog module, so the toolchain never reads VHDL itself; Yosys synthesizes it with synth\_xilinx -family xc3se and ABC. Three optimization goals, chosen in the process properties as in ISE, decide how hard the flow works: Area, the default of a new project, keeps the smaller of the usual mapping’s netlist and the mappings for speed, and places once; Speed uses the mappings for speed and makes several placements (32 for a design of up to 100 slices, 8 up to 500, 4 above), each routed, and keeps the fastest by Silinx’s timing analysis; Fast uses the usual mapping and one placement, except that a deep design (more than 16 logic levels, at most 1 500 LUTs) also gets Speed’s synthesis, kept if it is 10% faster.

![](images/ca324d46a75ea9e5e699a6bf5f881ab610414f4ddc32192b2072419217459611.jpg)  
Fig. 2. The open implementation flow for the Spartan-3E and Spartan-6. Every stage runs in the browser (in Web Workers) or in Node.js; no Xilinx tool is used. The iCE40 uses Yosys, nextpnr-ice40 and icepack.

## B. The bit database

The bitstream writer needs to know which configuration bits set each feature of the chip: each LUT memory bit, each multiplexer setting inside a slice, each I/O setting and each routing switch (PIP). Following IceStorm, Trellis and X-Ray, Silinx builds this database by observing the vendor’s tools. Small test designs that difer in a single feature are written in XDL, ISE’s documented text design format; ISE’s command-line tools turn them into bitstreams (xdl -xdl2ncd, then bitgen), and the bits that change between two bitstreams are those of the feature that changed. The packets, registers and frames follow Xilinx’s public documentation (UG332, XAPP452); for the XC3S250E, the configuration data is one write of 578 frames of 73 32-bit words. The other Spartan-3E devices share the tile types’ bits with the XC3S250E; the Spartan-6 database was measured the same way. Every experiment and analysis script is kept in the repository, so that anyone with a lawful copy of ISE can rebuild and check the database.

No Xilinx program is decompiled or disassembled, no internal data file of ISE is read, no Xilinx file is shipped, and no protection (bitstream encryption, readback protection, licence checks) is studied. The method relies on the right of a lawful user to observe, study and test how a program functions (EU Directive 2009/24/EC, Art. 5(3)); which bit configures which feature is a fact about the hardware’s interface rather than an expression of ISE’s code.

## C. Device description and timing

Placement and routing need the chip’s tiles, sites, wires and switches. Silinx computes a compact description from the output of ISE’s documented xdl -report -pips -all\_conns command. For the XC3S250E the report is 204 MB of text: 1 419 tiles, 4 615 sites, 569 094 tile wires and 2 139 368 PIPs. Since most tiles of a kind are identical, the description stores each distinct tile content once (293 templates; the 576 CLB tiles share one) and the connections between tiles as 200 181 nodes; gzipped, it is 1.96 MB. The descriptions of the six devices of the supported boards are shipped, so the flow needs no ISE; the four Spartan-3E dies have also been measured by black-box observation, and designs routed on the measured descriptions pass ISE’s DRC with byte-identical bitstreams. nextpnr’s chip database is generated from these descriptions, restricted to the PIPs whose bits are known, with Silinx’s delay model, which is fitted to ISE’s timing analyser (trce).

## D. Packing, placement and routing

nextpnr packs, places and routes, timing-driven, at every goal; on the Spartan-6, Silinx’s own packer runs first. The routed design is written back as XDL, so the bitstream writer and the ISE oracle of Section V see the same format as before. Until late October 2026, placement and routing were done by Silinx’s own simulated-annealing placer [14] and PathFinder-style router [15]; nextpnr became the default in release 15.21.0, and the only engine in 15.22.0, once it had reached parity, while the Spartan-6 packer and the bitstream writer remain Silinx’s.

## V. Verification: ISE as the Oracle

An open flow for a closed device is only useful if its bitstreams are right: a wrong bit can short two drivers on a student’s board. Silinx therefore treats ISE as the reference at every level.

Byte-identical bitstreams: The design placed and routed by Silinx is written out as XDL and given to ISE, which converts it (xdl -xdl2ncd), runs its DRC and writes its own bitstream with bitgen. Silinx’s bitstream for the same design must be byte-identical to ISE’s, and ISE’s DRC must find no error (npm run check:open-flow). This holds for the designs checked on the Spartan-3E and on the Spartan-6 boards. Negative controls show that the check is meaningful: a design with all its routing removed gives 372 DRC errors, and a single PIP leading nowhere gives an “antenna” error.

The examples are the test designs: Ten example projects form the course’s progression and the regression suite: gates (a schematic), trafic (a state diagram, a Boolean function, VHDL and Verilog), blinky (an ASM chart), pong, stopwatch, clock, vga, isqrt (a module written in C, made into hardware by Silinx HLS), uart and micro80 (a microprocessor running a program from block RAM). Each must build and run on every device of the open toolchain and build with ISE with no warnings, and each has a self-checking test bench.

Tests and continuous integration: Every change ships with its own tests, and tests assert the exact expected warnings of each flow, never only the absence of errors. A pull request runs the unit tests on Linux and the UI tests in both interfaces and in Firefox and WebKit; the ISE tests run nightly and the comparison with ISE weekly, on a self-hosted runner with a private ISE installation that is never published.

## VI. Evaluation

Setup: The design corpus collects 67 open-source designs: CPUs (T80, light8080, PicoRV32, NEORV32, SERV, FemtoRV32), peripherals (UART, SPI, I2C, FIFOs), cryptographic cores (AES, SHA-1, SHA-256, SipHash), floating-point units and benchmarks (EPFL [16], ITC’99, IWLS’05, VTR [9]). Each design is built by both flows for the same chip, the XC3S1200E-FG320 (Nexys2-1200), with the same top, at each of Silinx’s three optimization goals, and by ISE at the matching XST setting: Area against -opt\_mode Area -opt\_level 2, Speed against -opt\_mode Speed -opt\_level 2, and Fast against -opt\_mode Speed -opt\_level 1, XST’s default. ISE’s map and par are given their best chance at every goal: map packs slices as tightly as it can (-c 1) and the clock constraint is set to 5 ns, so that its place and route works for speed.

## TABLE I

Silinx vs. ISE per optimization goal on the XC3S1200E: geometric means of the ratios Silinx/ISE over the 45 designs built by both flows (critical path: over those timed by both). Negative: Silinx smaller, faster or quicker.
<table><tr><td>ISE (XST) setting</td><td>Area Area, 2</td><td>Speed Speed, 2</td><td>Fast Speed, 1</td></tr><tr><td>Logic LUTs</td><td>-16%</td><td>-21%</td><td>-19%</td></tr><tr><td>Flip-flops</td><td>+5%</td><td>-9%</td><td>-4%</td></tr><tr><td>Slices</td><td>-15%</td><td>-21%</td><td>-20%</td></tr><tr><td>As few LUTs or fewer</td><td>30/45</td><td>33/45</td><td>26/45</td></tr><tr><td>Critical path</td><td>-14%</td><td>-8%</td><td>+12%</td></tr><tr><td>As fast or faster</td><td>29/39</td><td>28/40</td><td>10/39</td></tr><tr><td>Run time (indicative)</td><td>-80%</td><td>-11%</td><td>-84%</td></tr></table>

TABLE II

Selected designs: ratios Silinx/ISE of logic LUTs (L) and critical path (CP) at each goal.
<table><tr><td rowspan="2">Design</td><td colspan="2">Area</td><td colspan="2">Speed</td><td colspan="2">Fast</td></tr><tr><td>L</td><td>CP</td><td>L</td><td>CP</td><td>L</td><td>CP</td></tr><tr><td>vtr-mkPktMerge</td><td>0.06</td><td>0.64</td><td>0.06</td><td>0.80</td><td>0.06</td><td>0.85</td></tr><tr><td>light8080</td><td>0.37</td><td>0.78</td><td>0.36</td><td>0.83</td><td>0.37</td><td>0.78</td></tr><tr><td>epfl-adder</td><td>0.57</td><td>0.13</td><td>0.39</td><td>0.86</td><td>0.25</td><td>1.23</td></tr><tr><td>epfl-priority</td><td>0.46</td><td>1.82</td><td>0.51</td><td>0.41</td><td>0.42</td><td>2.07</td></tr><tr><td>epfl-voter</td><td>0.65</td><td>0.81</td><td>0.56</td><td>0.84</td><td>0.56</td><td>0.96</td></tr><tr><td>fpu-adder</td><td>0.70</td><td>0.93</td><td>0.70</td><td>0.86</td><td>0.70</td><td>0.96</td></tr><tr><td>t80</td><td>1.00</td><td>0.86</td><td>0.97</td><td>0.99</td><td>1.15</td><td>1.29</td></tr><tr><td>secworks-siphash</td><td>1.10</td><td>0.98</td><td>1.03</td><td>1.12</td><td>1.04</td><td>1.10</td></tr><tr><td>itc99-b05</td><td>1.28</td><td>1.10</td><td>1.17</td><td>1.06</td><td>1.22</td><td>1.30</td></tr></table>

Both critical paths are measured by ISE’s trce on the routed design. The results were measured on 8 November 2026 (Silinx 15.22.1, ISE 14.7); the full tables are in the repository (docs/QUALITY.md).

Size: Table I summarises the comparison. Of the 67 designs, 45 were built by both flows at each goal. At every goal the open flow uses fewer logic LUTs and slices than ISE at the same goal (15–21% in geometric mean), and as few LUTs or fewer on 26 to 33 of the 45 designs. The ratios vary widely between designs (Table II): at Area, the slices of vtr-mkPktMerge are 0.12 of ISE’s, those of itc99-b05 2.10 times ISE’s.

Speed: At Area and Speed the open flow’s critical paths are shorter than ISE’s at the same goal, by 14% and 8%, and as short or shorter on 29 of 39 and 28 of 40 designs. At Fast, which makes one placement of the usual mapping, they are 12% longer, and as short or shorter on only 10 of 39. Individual designs show why the goals matter: epfl-adder, a 128-bit adder written as gates, has a critical path of 15.0 ns at Area against ISE’s 113.3 ns, while epfl-priority at Area is 1.82 times slower than ISE and 0.41 times at Speed. Splitting each critical path that is longer than ISE’s into logic and routing, as trce reports it, the average excess at Area (10 designs) is 1.76 ns of logic and 1.15 ns of routing, and at Fast (29 designs) 2.30 ns and 1.28 ns; at Speed (12 designs) the logic is on par (−0.10 ns) and the excess is in routing (+1.00 ns), which is where further work lies. For the circuits of an introductory course, which are small and meet the boards’ 50 MHz clock with a wide margin, these diferences matter little.

Run time: Each flow’s whole build, from synthesis to bitstream, was timed on its own machine, Silinx’s in WebAssembly, so the run times are an indication only: in geometric mean, Silinx’s build took 80% and 84% less time than ISE’s at Area and Fast, and 11% less at Speed, which places and routes several times.

Language coverage: XST, ISE’s synthesis tool, cannot read 15 of the 67 designs (\$clog2, VHDL-2008 constructs such as reading an output port, libraries other than work, some for loop forms, event expressions in sensitivity lists); Silinx reads and builds them. Seven more designs were not built by ISE because it rejected their pin constraints. For students, this means that HDL idioms from current textbooks and online examples work.

History: The comparison has been run since mid-October 2026. With Silinx’s earlier in-house placer and router, the last comparison at the default efort (25 October, Silinx 15.13.0) gave 20% fewer LUTs and 7% shorter critical paths than ISE.

## VII. Developing with AI Coding Agents

Silinx was written with Anthropic’s Claude Code: the author sets the direction, files the issues, tries the tool on the boards and reviews; Claude sessions write the code, the tests and the documentation. The name says so: Si, for the AI that wrote it, and linx, for the programmable links of an FPGA. Several sessions work in parallel, each in its own git worktree on its own issue, with one session coordinating the merges.

The agents follow written rules, kept in the repository and read by every session at start-up: every bug, feature or to-do is a public issue, and the fixing change refers to it; every change ships with its own tests (unit tests compared against an independent reference where possible, UI tests in both interfaces, generated HDL compiled and simulated in VHDL and in Verilog, a regression test that fails on the old code); tests assert the exact warnings; modules are kept small, so that a session reads only what it changes; the examples are the test designs; and ISE is the oracle. Changes reach the protected main branch only through pull requests with the continuous integration checks green, reviewed by a code owner. An external, strict oracle suits AI-generated code: a bitstream is either byte-identical to ISE’s or it is not, and a failing comparison cannot be argued away. From its first version on 6 October 2026 to 8 November 2026, the repository’s main branch received 2 326 commits, its issues and pull requests reached number 3 141, and 50 releases were published.

## VIII. Limitations and Future Work

The open toolchain supports the Spartan-3E boards (Basys2, Nexys2), the Spartan-6 boards (Nexys3, Atlys)

and the iCEBreaker; other boards still need ISE through the local server, and further families are planned. The Spartan-6 flow does not yet cover clock managers’ config urations beyond those measured. Speed results are below ISE’s at the Fast goal, and the remaining critical-path gap at Speed lies in routing. The shipped device descriptions are still the ones computed from an ISE report; the measured descriptions of the Spartan-3E dies, which avoid that dependence, pass the same checks but do not yet replace them. The quality comparison covers the Spartan-3E only. Finally, this paper describes the platform and its technical validation; Silinx is used in the author’s courses, but a study of its use in the classroom, and of its efect on students’ learning, is future work.

## IX. Conclusion

With Silinx, Spartan-3E and Spartan-6 teaching boards stay in use, and other boards are supported too: students design in HDL, in C or C++, or by drawing, then simulate, emulate, synthesize, place, route and program a Basys2, Nexys2, Nexys3, Atlys or iCEBreaker from a web page, on the computer or tablet they already have, and the Xilinx bitstreams are checked bit for bit against the vendor’s own tools. Silinx is free software (AGPL-3.0) at https://github.com/pmnmalo/Silinx and runs at https://pmnmalo.github.io/Silinx/.

## Acknowledgment

Silinx is an independent project, not afiliated with or endorsed by AMD/Xilinx or Digilent. Xilinx, ISE, ISim, iMPACT, Spartan and Vivado are trademarks of AMD; Basys and Nexys are trademarks of Digilent. The code of Silinx was generated with Claude Code (Anthropic); the author directed and reviewed it and takes responsibility for this paper.

## References

[1] J. Arroyo Torrens et al., “Icestudio: a visual editor for open FPGA boards,” https://github.com/FPGAwars/icestudio.

[2] C. Wolf and M. Lasser, “Project IceStorm,” https://github. com/YosysHQ/icestorm.

[3] D. Shah et al., “Project Trellis: documenting the Lattice ECP5 bit-stream format,” https://github.com/YosysHQ/prjtrellis.

[4] “Project X-Ray: documenting the Xilinx 7-series bit-stream format,” https://github.com/f4pga/prjxray.

[5] C. Wolf, “Yosys Open SYnthesis Suite,” https://yosyshq.net/ yosys/.

[6] D. Shah, E. Hung, C. Wolf, S. Bazanski, D. Gisselquist, and M. Milanovic, “Yosys+nextpnr: an open source framework from Verilog to bitstream for commercial FPGAs,” in Proc. IEEE FCCM, 2019, pp. 1–4.

[7] “nextpnr-spartan: Spartan-3E and Spartan-6 microarchitectures for nextpnr Himbaechel,” https://github.com/ pmnmalo/nextpnr-spartan.

[8] “YoWASP: Unoficial WebAssembly-based packages for Yosys, nextpnr, and more,” https://yowasp.org/.

[9] K. E. Murray et al., “VTR 8: high-performance CAD and customizable FPGA architecture modelling,” ACM Trans. Reconfigurable Technol. Syst., vol. 13, no. 2, 2020.

[10] J.-B. Note and É. Rannaud, “From the bitstream to the netlist,” in Proc. ACM/SIGDA FPGA, 2008, pp. 264–274.

[11] F. Benz, A. Sefrin, and S. A. Huss, “BIL: a tool-chain for bitstream reverse-engineering,” in Proc. FPL, 2012, pp. 735– 738.

[12] N. Steiner, A. Wood, H. Shojaei, J. Couch, P. Athanas, and M. French, “Torc: towards an open-source tool flow,” in Proc. ACM/SIGDA FPGA, 2011, pp. 41–44.

[13] C. Lavin, M. Padilla, J. Lamprecht, P. Lundrigan, B. Nelson, and B. Hutchings, “RapidSmith: do-it-yourself CAD tools for Xilinx FPGAs,” in Proc. FPL, 2011, pp. 349–355.

[14] A. Marquardt, V. Betz, and J. Rose, “Timing-driven placement for FPGAs,” in Proc. ACM/SIGDA FPGA, 2000, pp. 203–213.

[15] L. McMurchie and C. Ebeling, “PathFinder: a negotiation-based performance-driven router for FPGAs,” in Proc. ACM FPGA, 1995, pp. 111–117.

[16] L. Amarù, P.-E. Gaillardon, and G. De Micheli, “The EPFL combinational benchmark suite,” in Proc. Int. Workshop on Logic and Synthesis (IWLS), 2015.