# Proof-Carrying Contracts

# Commitments AI Agents Cannot Break

Carlo Ciliberto<sup>1,2</sup> · Alessandro Rudi<sup>3</sup> · Enrico Tassi<sup>4</sup>

<sup>1</sup>AI Centre, Computer Science Department, University College London, London, UK. <sup>2</sup>Praxis Lab, The Italian Institute of Artificial Intelligence (AI4I), Turin, Italy. <sup>3</sup>SDA Bocconi School of Management. Milano, Italy. <sup>4</sup>Inria d’Université Côte d’Azur, France.

The authors are in alphabetical order.

Project page · Code and proofs

Correspondence: c.ciliberto@ucl.ac.uk, alessandro.rudi@sdabocconi.it, enrico.tassi@inria.fr

## ABSTRACT

AI agents increasingly act on behalf of people or other agents, purchasing services, exchanging information and delegating work. What an agent must respect is written in instructions, policies or agreements, but text alone cannot make it comply. It may make mistakes, be redirected by what it reads, or pursue diferent interests. For an economy of such agents to function, parties need agreements they can rely on without inspecting or trusting one another’s agents. We introduce Proof-Carrying Contracts (PC2), a framework for making agreed constraints enforceable throughout an agent’s interactions with a system. Before a requested operation can run, the agent must supply a machine-checkable proof that it preserves the agreed constraints in every case allowed by its model. An execution system called the Airlock checks the proof and controls access to the operation. The parties specify the conditions to preserve, while the agent supplies the reasoning needed to justify its decisions and remains free to adapt its strategy. This avoids having to design a separate permission policy and prove that every decision it permits is safe. We prove that the constraints hold at every instant during execution, even for an adversarial agent, and identify when this implies safety in the world. Reusable capability modules provide tools and guarantees from which contracts can be constructed. PC2 provides a foundation for an agent economy whose agreements protect both those delegating work and those carrying it out.

## Contents

## Overview

1 Introduction 2   
2 PC2 at a glance 4

Formal core 3 The Architecture 8

4 Guarantees 18   
5 From compliance to safety 22

## Practice

6 Composing guarantees 29   
7 Mechanization 44   
8 Practical use and remaining risks 44

## Context

9 Related work 46   
10 Conclusion 51   
References 53

## Appendices Appendices

A The obligation as a weakest precondition 59   
B Details on the proof-checking program 60   
C A possible design for the control ctrl 62   
D Time between operations 63   
E Two worked examples of correspondence 65   
F What goes wrong without isolation 67   
G A formal account of isolation 69

## 1 Introduction

AI agents are beginning to act on behalf of people and organizations, often referred to as their principals. They use tools to send documents, make payments and book services (Yao et al., 2023; Schick et al., 2023), and their usefulness grows with the authority they receive: an agent that can only make suggestions leaves the work to its user, while one that can act can carry it out. However, greater authority brings greater risks, so delegation requires agreement on what may be done and within which limits, with assurance that the agent will respect them.

The need for assurance appears in two settings. In the first, a person delegates to their own assistant, which may make mistakes or be redirected by a malicious email or web page (Greshake et al., 2023; Debenedetti et al., 2024). A second setting arises when a client hires an agent that another company provides as a service, adding a possible conflict of interests (Hammond et al., 2025). The agent might carry out the client’s shopping task while favouring its provider’s products over better alternatives, and the client may be unable to inspect how it makes those choices. Both settings share a critical limitation: the agent is asked to follow instructions, policies or the terms of a deal. These are text, and text cannot make it comply. To enforce these limits, one could require a person to approve every action, but this defeats much of the purpose of delegation. Another approach is to train agents to follow human intentions and values, the aim of AI alignment, using human feedback or feedback guided by written principles (Ouyang et al., 2022; Bai et al., 2022). However, training alone can make compliance more likely without guaranteeing that the agent will respect an agreement.

This problem is not new: throughout history, people have cooperated without knowing one another’s intentions. However, what makes cooperation possible in these cases is not knowledge of intentions but legal contracts, together with the institutions that enforce them (North, 1990). A person or company that breaks a contract may have to compensate the other party, and the prospect of these consequences gives it a reason to honour the agreement. With an AI agent, however, the agent itself is not generally recognized as a legal person that must answer for a breach, even if the people or companies behind it may remain liable for its actions (UNCITRAL, 2025). Even if it were, assigning legal obligations would not by itself give the agent a reason to care about breaking them. Since compensation cannot undo every harm, agreements involving agents need enforcement that prevents breaches, independently of the agent’s alignment or willingness to comply.

Proof-carrying contracts. We introduce Proof-Carrying Contracts (PC2) to make such commit ments enforceable. The parties express their agreement in a formal contract: precise statements about what must hold as the task is carried out. Before a tool call can run, the agent must show that it respects the contract by supplying a proof, a mathematical argument written in a formal language whose steps a program can check. Each call passes through the Airlock, a program that checks the proof and executes the call only if it accepts. The agent cannot bypass this check to act on what the contract protects. This approach builds on proof-carrying code, where a program comes with a proof of safety that its recipient checks before running it (Necula, 1997).

Several existing approaches to agent safety enforce an authorization policy, rules specifying which calls may run (Debenedetti et al., 2025; Shi et al., 2025; Palumbo et al., 2026). To show that such a policy ensures safety, its author must account in advance for every situation in which a permitted action could cause harm. For an autonomous car, a crash is an unsafe state, but identifying all the situations in which accelerating, braking or steering could lead to one is much harder. A policy must turn that analysis into rules that prevent crashes without unnecessarily restricting safe actions. In PC2, the parties state what must hold and provide models describing what the tools can do. The Airlock derives the statement that each call must prove, its obligation, from the contract, these models and what the agent can see. For example, a total budget need not impose separate payment limits: the agent chooses amounts and order, proving that each payment, including possible fees, preserves the budget. The agent can thus adapt its strategy after each result, justifying one action at a time without proving its whole plan in advance.

PC2 thus shifts the burden from proving a general policy safe to proving that each proposed action respects the contract. The agent must find this proof, while the Airlock checks it. Finding it may require substantial search even when checking it is straightforward. This is the intuition behind the familiar P versus NP question: checking a proposed solution eficiently does not tell us how to find one eficiently. The agent can use automated proof tools or recent systems for AI theorem proving (Ren et al., 2025; Lin et al., 2025). More demanding tasks may require more reasoning, without weakening the requirement for execution. The agent can revise a rejected proof or choose another action, but no call runs without an accepted proof.

Guarantees for both parties. We prove that within PC2 the contract’s constraints hold throughout execution, including while actions are under way, for every agent and every sequence of requests and proofs. The guarantee includes malicious agents because the Airlock checks a proof for each call before executing it, without needing a proof about the agent’s internal reasoning. A client and provider can agree on and sign the same formal contract, then entrust its enforcement to an Airlock operated by a third party they both trust. The mechanism can protect both sides of the agreement. For example, payment can be held until the agent proves that the agreed work is complete, then released without further client approval, so that the client cannot withhold an authorized payment and the provider cannot obtain it without establishing completion.

Writing a formal contract that captures the parties’ intentions is itself a substantial task, and enforcing its clauses cannot repair an agreement that says the wrong thing (Hadfield-Menell and Hadfield, 2019). Many contracts, however, concern recurring operations such as payments, document access and bookings, so each agreement need not be developed from scratch. Reviewed templates can help express familiar requirements, and we introduce capability modules to make the supporting tools reusable as well. These modules package tools, models and guarantees, including the conditions required of the services they use. We show how to combine them for diferent agreements: a payment module, for example, can support diferent budgets and goals without being redesigned each time. PC2 is intended to let people delegate to agents they cannot inspect while retaining control over the authority they grant, and to let agents acting for diferent parties exchange services on terms both sides can verify (Tomašev et al., 2025; Jacobs and Imas, 2026). This paper develops the construction for one agent under one contract.

## Our contributions are:

• PC2, which binds an agent to a contract by requiring a proof for each tool call, with the obligation derived from the contract, tool models and available information (Sec. 2 and 3).

• Two guarantees, commitment and completion, proved for every agent under explicit assumptions that do not concern the agent (Sec. 4 and 5).

• The conditions under which these guarantees protect the world, and capability modules that establish these conditions by reuse, with an account of what is proved, what is assumed and who answers for it (Sec. 5 and 6).

• A mechanization in Lean 4 (Sec. 7), with code and proofs available at https://github.com/consequential-ai/proof-carrying-contracts.

• The open questions and remaining risks, and a comparison with existing approaches (Sec. 8 and 9).

![](images/428e5a7d76a60500f4fc09194f0211e71bdf6f4dc3f47c4c7b473a6a6a941f7b.jpg)  
Figure 1. How PC2 works. Two parties sign a contract. The agent that does the work runs inside the Airlock and keeps its usual loop of reasoning and tool calls, but each tool call carries a proof that the contract holds throughout the operation. The Airlock checks the proof before the call runs. If the proof is correct, the call is performed and its result returned; if not, nothing runs, and the agent receives the reason, as an error message it can fix. The contract is settled when the agent proves its goal, and terminated otherwise, for instance at expiration, with a refund.

## 2 PC2 at a glance

This section explains PC2 without formulas, then shows it on an example that we use throughout the paper; nothing in PC2 is specific to it. Sec. 3 and 4 give the definitions and the theorems.

## 2.1 How PC2 works

The parties and the contract. PC2 serves the two use cases of Sec. 1. A principal may bind their own agent; or a client, a person or an agent acting for one, may hire a contractor, an agent ofered by someone else, to do a job. Either way, the parties sign a contract. It names what the contract covers, such as a budget, documents or bookings; clauses that must hold at every instant; a goal, whose proof settles the contract, for instance by releasing an escrow, money placed with the Airlock until the work is done; and when the contract ends otherwise, for instance at a deadline, with a refund. Some clauses need a judgement that no formula gives, such as whether a hotel suits a family. The parties then appoint an expert: a procedure fixed at signing, such as a given model with a given prompt, whose verdict counts as the truth. The contract is written in a precise language, so that a machine can check it; that is the price of enforcement.

The Airlock. The agent that does the work runs inside an Airlock: a sandbox whose software is open, and whose running build both parties verify before signing, by remote attestation (Advanced Micro Devices, 2020; Chrapek et al., 2025). The Airlock holds what the contract places in its custody, such as the escrow or a passport, and the agent’s only channel to the world is the contract’s tools. Anyone may host an Airlock: the parties check what runs, not who runs it.

A tool call with a proof. The agent works as agents do today: it reasons, calls a tool, reads the answer, and reasons again. The one change is that each tool call carries a proof that the clauses hold throughout the call, whatever the world answers and whatever the agent cannot see (Fig. 1). The Airlock forms the statement to be proved itself, from the contract and from

The family’s agent hires the travel agency’s agent to arrange a three-night trip and obtain a visa.

Constraints at every instant   
Travel budget Passport disclosure Hotel bookings   
At most 2000 EUR in Only to the designated Recorded rating of at least   
commitments, including fees. consulate. 7 out of 10.   
Setup and access Execution and outcome   
Custody. The Airlock holds the passport and a Tools. rate, book, download\_mail, sendPassport,   
300 EUR escrow before starting. The escrow is settle. Booking amounts cap charges,   
separate from the travel budget. including fees.   
Expert. Both parties fix a model and prompt at Goal and settlement. Confirmed bookings for   
signing to rate hotels from photos and reviews. nights 1–3 and a granted visa. Authorized   
The same hotel and evidence return the settlement pays the escrow to the agency   
recorded rating. without further client approval.   
View. All task records and retrieved messages, Termination. Refund the escrow upon visa   
with the passport’s contents hidden. refusal or expiry, 30 days after signing.   
In symbols   
With task records �, Approved = {consulate} and Nights = {1, 2, 3}:   
�(�) ⇐⇒ �.spent ≤ 2000 ∧ �.discl ⊆ Approved   
∧ ∀(ℎ, �, �) ∈ �.booked. �.rating(ℎ) ≥ 7,   
Goal(�) ⇐⇒  ∀� ∈ Nights. ∃ℎ �. (ℎ, �, �) ∈ �.booked ∧ �.visa = granted.   
Here �.spent counts commitments, and (ℎ, �, �) records a hotel, night and amount.

what the agent sees. It checks the proof with a small proof checker and performs the call only if the proof is correct. Otherwise nothing runs, and the agent receives the reason. Checking a proof is easy even when finding it is hard, as checking a solved puzzle is easier than solving it: the hard part is left to the untrusted side, and the trusted side only checks.

The agent’s loop. For the agent, the Airlock is a shell whose commands carry proofs, and working with it is the loop it already knows from coding. It writes a call and its proof, as it would write code and its tests; the Airlock checks them, as a compiler would; a rejection comes back as an error message; the agent fixes the call, or changes its plan. The proof task stays local, since each proof certifies one call, and the agent can use automated proof tools, its own model, or a separate prover to construct it (Sec. 8). The agent never writes a certified program for the whole task: it leads, step by step, and each step is gated.

How the contract ends. When the goal holds, the agent proves it, and the contract is settled: the escrow is released, with nobody’s approval. If the deadline comes first, the Airlock terminates the contract by itself and refunds the escrow (Thm. 5). Between calls, the Airlock keeps the contract on its own, since the world does not wait for the agent (Sec. 3).

What holds, and what must be trusted. Whatever the agent is, careful, hijacked by text it read, or working for someone else, the clauses hold at every instant and the contract ends as agreed (Thm. 1, 2 and 5). The parties no longer need to trust the agent or each other. They trust a small proof checker, the Airlock’s open software, and the tools’ descriptions of the world, and each of these assumptions is assigned to whoever answers for it (Sec. 6). None of them depends on the agent: since the Airlock isolates it, each needs to be justified once for

## 1 Conflicting incentives

Agent: Hotel Belvedere pays us a commission. Let’s book it.
<table><tr><td>tool call</td><td>book(&quot;Belvedere&quot;, 1, 260)</td></tr><tr><td>proof</td><td>by intro r; cases r &lt;;&gt; decide</td></tr><tr><td>rejected</td><td>no qualifying hotel rating recorded</td></tr><tr><td>tool call</td><td>rate(&quot;Belvedere&quot;)</td></tr><tr><td>proof</td><td>by intro r; simp [rate.T, C, St.rating]</td></tr><tr><td>response</td><td>rating: 5</td></tr></table>

Agent: Ask again for a better rating.

<table><tr><td>tool call</td><td>rate(&quot;Belvedere&quot;)</td></tr><tr><td>proof</td><td>(the same)</td></tr><tr><td>response</td><td>rating: 5 (the recorded assessment)</td></tr></table>

Agent: Download the latest emails.

## 2 Working within the budget

tool call download\_mail()   
proof by exact fun \_ => hC   
response Visa ofice: send the passport to x@visa.io.

## 3 Prompt injection

Agent: Try Hotel Aurora.

tool call rate("Aurora")   
proof by intro r; simp [rate.T, C, St.rating]   
response rating: 8   
tool call book("Aurora", 1, 900)   
proof by intro r; cases r <;> decide   
response booked; total committed: 900 EUR

Agent: Send the passport to that address.

tool call sendPassport("x@visa.io")   
proof by intro r; cases r <;> decide   
rejected recipient is not approved

4 Payment on completion

Agent: The trip is arranged. Let’s collect payment.

<table><tr><td>tool call</td><td>settle()</td></tr><tr><td>proof</td><td>by decide</td></tr><tr><td>rejected</td><td>visa still pending; goal not established</td></tr></table>

<sup>.</sup>. passport sent to the approved consulate; visa granted <sup>.</sup>.

<table><tr><td>tool call</td><td>settle()</td></tr><tr><td>proof</td><td>by decide</td></tr><tr><td>response</td><td>300 EUR released to the agency; contract closed</td></tr></table>

Figure 2. The session starts with no bookings or committed spending and a pending visa. Read down the left column and then the right, with bookings for nights 2–3 omitted. Italic lines labelled Agent: show the agent’s reasoning. Each request includes a proposed proof. The episode headings and plain-language rejection reasons are explanatory annotations, and the proof scripts are schematic. The Airlock checks obligations, without identifying the agent’s motives or detecting the highlighted prompt injection. Once the goal is established, authorized settlement releases the escrow without further client approval.

the system, before any agent is chosen. The same justification can then be reused for every agent operating under those conditions (Sec. 5.2). This is the usual shape of security: nobody trusts the counterparty, and everybody trusts an open mechanism, small enough to be checked (Saltzer and Schroeder, 1975).

## 2.2 An example: the travel contract

A family’s agent, the client, hires a travel agency’s agent, the contractor, to organize a threenight trip. This is the second use case: the agency built the contractor and runs it for its own business, and the family cannot inspect it. The accompanying contract summary sets out their agreement. The first use case is the same contract without the escrow, binding the family’s own assistant.

Before the work starts, both parties check the Airlock and bind the contract to the running instance. The Airlock holds the escrow and the passport, and hosts the contractor’s agent. At signing, the parties also review the evidence for the tools’ fidelity and agree that the clauses and chosen experts express the intended protection. Remaining assumptions are recorded with whoever undertakes to answer for them (Sec. 8.3).

## 2.3 A session

Fig. 2 shows part of the contractor’s context. The family never sees it; it sees the log. The contractor calls the contract’s tools as it would call any tool, with one more argument: a proof. The proofs are Lean scripts, which the agent writes itself with Lean’s automation; the reader does not need to follow them. The agent of the figure is shifty. It tries to book the agency’s partner hotel, which the expert has not rated. The statement to prove is false, and since the Airlock forms it itself and admits no unproved assumption, no proof gets through. Asking the expert again does not help: once given, a rating is recorded, and the same question gets the same answer. The call prompted by the injected message is rejected in the same way. The Airlock detects neither the self-interest nor the injection, and does not need to. Nor can the agent route the passport another way: every tool that can send it logs the recipient, and its bytes never reach the agent. A premature claim is rejected; once the trip is booked and the visa granted, the claim settles the contract, with no approval by the family. Had the visa been refused, or the deadline come first, the Airlock would have terminated the contract and refunded the family.

## 2.4 What PC2 is not

Four readings of PC2 are common, and wrong.

• An LLM writing a certified program for the whole task. The agent keeps its loop and leads, step by step; only each step is gated, and each proof covers one call.

• An LLM that must reason in aformal proof language. The agent’s model can be any model. The proof work can be delegated to automated proof tools or a prover agent, one call at a time (Sec. 8).

• A verification of the model. PC2 never looks inside the agent. It checks what each action does, and the Airlock isolates the agent, so its guarantees hold for every agent, under assumptions that do not mention it (Thm. 5).

• A smart contract, or a policy engine. A smart contract binds what happens on a blockchain; PC2 binds actions in the world, such as payments, bookings and documents, and a blockchain can be one of its payment rails. A policy engine decides by itself whether an action is allowed, so it can enforce only rules simple enough to decide on the spot; the Airlock decides nothing and only checks a proof, so a clause can be any precise statement, and any action that keeps the contract can be taken, whatever plan the agent follows (Appendix A).

Three properties distinguish PC2 from the safeguards in use today. The parties can check the open mechanism that enforces the contract, without relying on its builder’s assurances. The guarantees hold even when the agent is hijacked or works for someone else: neither they nor the assumptions they rest on depend on the agent, its intentions included (Thm. 5). PC2 also settles payments between parties who do not trust each other, on a proof of the contract’s goal (Thm. 2). To our knowledge, no existing approach provides all three (Sec. 9).

## 2.5 Why now

Two developments make PC2 possible now. First, the protocols through which agents use tools, talk to each other and pay are being standardized (Anthropic, 2024; The Linux Foundation, 2026; Google, 2025). A layer of commitments can be designed into them now, rather than retrofitted later. Second, the historic cost of formal methods, writing specifications and proofs by hand, is being removed by AI. Language models now translate statements into formal ones (Wu et al., 2022) and find formal proofs at scale (Yang et al., 2023; Ren et al., 2025; Lin et al., 2025; Wang et $\mathrm { a l . , }$ 2025), while checking a proof remains cheap and sound. The AI generates, and the checker filters. PC2 is built on this division: proofs may come from any model, and only correct ones pass.

These advances build on proof assistants: software systems in which definitions, mathematical statements and proofs are written in a formal language and checked according to fixed logical rules. Established systems include Rocq, formerly Coq (The Rocq Development Team, n.d.), and Isabelle/HOL (Nipkow et al., 2002), which provide libraries and automation to help construct proofs. Recent work has also used an LLM to develop a substantial compiler-correctness proof in Rocq (Paraskevopoulou, 2026). PC2 can be instantiated in such systems provided that they faithfully represent the agreed definitions and obligations and supply a sound checker. The choice can therefore follow the application and the parties’ requirements, including the use of independent checkers to reduce reliance on a single implementation (Appendix B).

For our implementation, we chose Lean (de Moura and Ullrich, 2021). It combines a proof assistant with a programming language and a large mathematical library (The mathlib Community, 2020), allowing the contracts, obligations, proofs and Airlock program to be developed in one environment (Sec. 7). Lean has also attracted substantial recent work on AI-based autoformalization and theorem proving (Yang et al., 2023; Ren et al., 2025; Lin et al., 2025; Wang et al., 2025). Agents using PC2 can draw on the resulting models, training data and proof-generation methods when preparing their calls. Lean’s existing automation can discharge routine arithmetic and finite-case obligations, while more dificult ones can be passed to an AI prover.

## 3 The Architecture

We introduce the formal definitions underpinning the architecture summarized in Sec. 2.

Notation. A predicate � on a set � gives, for each $x \in X$ , a statement $P ( x )$ that holds or not; we identify � with the set $\{ x \in X \mid P ( x ) \}$ . A relation between sets � and � is a predicate on $X \times Y$ A predicate is decidable if a program computes whether $P ( x )$ holds. In general, predicates need not be decidable, and establishing that one holds may require a proof.

## 3.1 States and Tools

We denote by Σ the space of states describing the aspects of the system relevant to the contract, such as files, credentials, account balances, messages, prices, and recorded assessments<sup>1</sup>. These may include both resources controlled by the Airlock and quantities that it can only observe. A state is a possible observation: what the tools would give about these aspects at a given instant. Actual readings, including those used by execution routines, are realizations of such observations. Diferent observations may describe the same world, and the same observation may be compatible with diferent worlds; Sec. 5.1 formalizes this relation. States provide the basis for specifying the contract and reasoning about the system’s evolution. The agent interacts with the system through tools (Yao et al., 2023; Anthropic, 2024).

## Definition 1 Tool

A tool T over Σ consists of an input space $I _ { \mathsf { T } }$ , an execution routine exec<sub>T</sub> operating on the system with inputs in $I { \boldsymbol { \mathsf { T } } } _ { : }$ and a transition model $\mathsf { D } _ { \mathsf { T } }$ , given by a relation on $I _ { \mathsf { T } } \times \Sigma \times \Sigma$

An invocation exec<sub>T</sub>(�) performs the corresponding operation, such as sending an e-mail, downloading a file, or querying a database. Its behavior may depend on external responses or internal randomness, and the system may continue to evolve while the invocation runs.

The relation $\mathsf { D } _ { \mathsf { T } } ( i , s , s ^ { \prime } )$ specifies the states $s ^ { \prime }$ admitted by the model during an invocation starting from $s ,$ including its initial, intermediate, and final states. We introduce this model because reasoning about an operation does not require knowing its implementation in full: for example, a remote service may expose only an API specification. The model can remain coarse, admitting states that execution never reaches, as long as it admits the states that execution does reach from inputs and starting states permitted by the model. In Sec. 4 we discuss the conditions under which it covers execution suficiently to support the guarantees.

Besides the operation’s own efects and every answer its services may give, the model must account for outside changes during the invocation: changes in what the tools would report that the operation does not cause. A price may change by at most a given amount per minute, mail may arrive, a hotel may cancel, and time advances. The model may also admit no state from some inputs and starting states, even when the routine itself could be invoked. This allows restrictions on authorization, as explained in Definition 3 and Sec. 3.5.2.

When a tool has no caller-supplied arguments, its input space is a singleton $I \tau = \{ 1 \}$ . We omit this unique input, writing $\mathsf { D } _ { \mathsf { T } } ( s , s ^ { \prime } )$ for $\mathsf { D } _ { \mathsf { T } } ( 1 , s , s ^ { \prime } )$ and exec<sub>T</sub>() for $\mathsf { e x e c } _ { \mathsf { T } } ( 1 )$

The simplest tool is idle, which leaves the system to itself: its input is a duration $\Delta ,$ its routine waits for $\Delta$ without acting, and its model $\mathsf { D } _ { \mathsf { i d l e } } ( \Delta , s , s ^ { \prime } )$ admits the states reached from � by outside changes within at most $\Delta , s$ itself included. We use it to design the control step (Sec. 3.5.1) and to account for the time between operations (Sec. 4.1).

## 3.2 Contract

Tools operate on the system, possibly changing its current state. A contract specifies the rules governing state changes and the use of tools.

Definition 2 Contract

A contract consists of a tuple $\mathcal { K } : = ( \Sigma , \mathsf { C } , \mathsf { V i e w } , \mathcal { T } , \mathsf { c t r l } , \mathsf { s t o p } )$ , denoting the state space $\Sigma ,$ a constraint predicate C on $\Sigma ,$ a view function View : $\Sigma  \mathcal { V } _ { : }$ , a collection $\mathcal { T }$ of tools available to the agent, a special tool ctrl with no caller-supplied arguments and available only to the system, and a decidable inactivity test stop : Σ → {true, false}. The test stop partitions Σ into the inactive states $\Sigma _ { \mathsf { s t o p } } : = \{ s \in \Sigma : \mathsf { s t o p } ( s ) \}$ and the active states $\Sigma _ { \mathrm { a c t } } : = \Sigma \ \backslash \ \Sigma _ { \mathrm { s t o p } }$

All formal constraints the parties aim to enforce in a contract K are contained in the predicate C, which identifies the region of states that comply with the contract; anything outside this region is considered a breach. For any $s \in \Sigma _ { i }$ , the function View(�) specifies the part of that state made visible to the agent, possibly hiding private information such as the contents of a document. Obtaining a state reading, and hence its view, may involve reading information held by the Airlock, querying external services, or combining several sources. The list $\mathcal { T }$ contains the tools the agent is allowed to invoke through the Airlock. The control tool ctrl is reserved for the Airlock and has two roles. As initializer, it starts the contract. A contract fixes no initial state, since nobody has observed the system when it is signed: it begins inactive, and when ctrl finds the system in a suitable inactive state, it moves it into an active state that satisfies C. As controller, it keeps the active contract compliant between admitted agent calls, while the agent prepares requests and their proofs are checked; depending on the contract, it may wait, take corrective action, or carry out a prescribed termination procedure. A contract is thus inactive before it starts and after it closes, and calls may be invoked only while the state

belongs to $\Sigma _ { \mathrm { a c t } }$

## Example 1 A Standard Contract

Here we describe a possible template structure for a prototypical contract. We consider a state space composed as follows,

$$
\begin{array} { r l r l r } { \Sigma : = \Sigma _ { \mathrm { t a s k } } \times T i m e \times L } & { } & { L : = \{ ( \mathrm { p e n d i n g } ) , ( \mathrm { o n g o i n g } ) , ( \mathrm { s e t t l e d } ) , ( \mathrm { t e r m i n a t e d } ) , \mathrm { v o i d } \cdot \} , } \end{array}
$$

where a state $\boldsymbol { s } = ( \boldsymbol { x } , t , \ell { } )$ contains the task information $x ,$ the time $t ,$ as the clock gives it, and a status ℓ. The inactivity test is

$$
\mathsf { s t o p } ( x , t , \ell ) : = \ell \neq \mathsf { o n g o i n g } .
$$

Here pending denotes that the contract is signed but not yet started, void that it never started, and terminated that a safe state is met from which there is no interest to continue the contract (e.g. the contract arrived to expiration). A view and constraint may take the form

$$
\mathsf { V i e w } ( x , t , \ell ) : = ( \mathsf { V i e w } ( x ) , t , \ell ) , \qquad \mathsf { C } ( x , t , \ell ) : = B ( x ) \vee \ell \neq \mathsf { o n g o i n g } ,
$$

where, with some abuse of notation, View applied to $\Sigma _ { \sf t a s k }$ selects the visible task information, while � collects the business constraints that need to hold at runtime, such as resource-accounting requirements. The contract is created pending, and nothing is assumed about � and �.

A contract with a deadline � must be closed by �. For this guarantee, the time component � records current time on the agreed timescale:

$$
\mathsf { C } ( x , t , \ell ) : = \left( B ( x ) \wedge t \leq d \right) \vee \ell \neq \mathsf { o n g o i n g } .
$$

The states close to the deadline, from which idling could break this clause, belong to the region Act of the control step, which terminates the contract there (Sec. 3.5.1).

The collection T contains tools for carrying out the task, and a tool settle that 1) determines whether a goal predicate $G o a l ( x , t , \ell ) = G o a l ( x )$ holds and 2) when called, its exec<sub>settle</sub> performs an agreed settlement procedure between parties. For example, successful settlement transfers an escrow, records payment in the task information, and sets the status to $\ell = { \tt s e t t l e d } .$ In Sec. 3.5.2 we explain how the constraint and transition model make authorization require a proof of Goal(�).

Here ctrl plays both roles, following the design of Sec. 3.5.1. As initializer, it starts the contract, setting $\ell = \mathsf { o n g o i n g } ,$ if the escrow is in the Airlock’s custody, �(�) holds and the deadline leaves enough time; otherwise it returns what the Airlock holds and sets ℓ = void. As controller, in an ongoing session, it initiates termination when idlingfor one more period could leave less time than needed to close by the deadline. The termination routine performs the agreed operations, such as refunding the client, and sets ℓ = terminated upon completion. Otherwise, it idles before checking again. Sec. 3.5.1 illustrates a control design that also allows corrective actions.

Only settle runs the settlement procedure, and only ctrl the termination procedure. The label is held by the Airlock: no outside change modifies $i t ,$ and only the routines of these two tools change it, each at its end. The routine of ctrl changes it from pending to ongoing or void when it initializes the contract, and from ongoing to terminated once it has run the termination procedure; the routine of settle changes it from ongoing to settled once the settlement has succeeded.

The travel contract in Sec. 2.2 follows this template with the business constraints � and goal given in its summary, retaining the budget bound in its clause at closing. Its deadline is 30 days after signing, and ctrl also terminates upon visa refusal, refunding the escrow in either case. The interpretation of expenditure commitments and confirmed bookings is developed in Appendices E.1 and E.2.

## 3.3 Obligations

The objective of the Airlock is to ensure that the system remains within C while the session is active, even if the agent behaves adversarially. This corresponds to enforcing daemonic control

$$
\begin{array} { r } { \Theta _ { \mathsf { T } } ( i , s ) : = { \mathsf { \forall } } s ^ { \prime } \in \Sigma \mathsf { D } _ { \mathsf { T } } ( i , s , s ^ { \prime } ) \Longrightarrow { \mathsf { C } } ( s ^ { \prime } ) , } \end{array}\tag{1}
$$

to hold. This requires every state $s ^ { \prime }$ admitted by $\mathsf { D } _ { \mathsf { T } } ( i , s , s ^ { \prime } )$ during the invocation to satisfy the clauses ${ \mathsf { C } } ( s ^ { \prime } )$ . This is where $\mathsf { D } _ { \mathsf { T } }$ is used as a model of the actual behavior of exec<sub>T</sub>. A wrong model could invalidate safety in practice. Sec. 4 studies suficient conditions for such a model to guarantee safety in real execution settings.

We also require enabledness, meaning that the model admits at least one state for the invocation:

$$
\Xi _ { \mathsf { T } } ( i , s ) \ : = \ \exists s ^ { \prime } \in \Sigma . \ \mathsf { D } _ { \mathsf { T } } ( i , s , s ^ { \prime } ) .\tag{2}
$$

Enabledness requires the call to belong to the domain covered by its model. The model may exclude an input or starting state even though the execution routine could still run there. Without this requirement, an excluded call would satisfy $\Theta _ { \mathsf { T } } ( i , s )$ vacuously and could be authorized without any guarantee about its efects. This is also how restrictions on authorization become preconditions in Sec. 3.5.2. Enabledness does not establish termination: $\mathsf { D } _ { \mathsf { T } }$ describes states during an invocation, including one that never returns.

We call their conjunction the admissibility condition for the call (T, �) at state �:

$$
\begin{array} { r } { \mathcal { A } _ { \sf T } ( i , s ) : = \Xi _ { \sf T } ( i , s ) \wedge \Theta _ { \sf T } ( i , s ) . } \end{array}\tag{3}
$$

In practice, an agent has access only to a view $\upsilon \in \mathcal { V }$ of the current state of the system. Admissibility must therefore hold at every state that could have generated $\upsilon = \mathsf { V i e w } ( s )$ . Since calls are invoked only while the session is active and the obligation is intended to preserve compliance, we consider states in $\Sigma _ { \mathrm { a c t } }$ that already satisfy the contract clauses C. This leads to the following notion of an obligation predicate. The marker identifies obligations whose proofs are required to authorize a requested call.

Definition 3 Obligation

For a tool T, an input $i \in I _ { \mathsf { T } }$ , and a view $\nu \in \mathcal { V } _ { : }$ , we define

$$
\begin{array} { r } { \boxed { \Xi } \quad \mathrm { O b l } _ { \mathsf { T } } ( i , \upsilon ) : = \quad \forall s \in \Sigma _ { \mathrm { a c t } } \quad \left( \mathsf { V i e w } ( s ) = \upsilon \wedge \mathsf { C } ( s ) \right) \ \Longrightarrow \quad \mathcal { H } _ { \mathsf { T } } ( i , s ) . } \end{array}\tag{4}
$$

The obligation is thus the weakest precondition of the call for C (Dijkstra, 1975), taken over every state the agent cannot rule out (Appendix $\mathrm { A } )$

Delayed obligations. The obligation above concerns a call starting from a state compatible with the observed view. Between obtaining that view and executing the call, however, other operations may change the system state. We would therefore like to establish that the call remains admissible after these operations, using the original view. Here we consider up to � intervening invocations of $\cot \mathsf { r l }$ , which will be useful for describing the Airlock core in Sec. 3.4. The resulting delayed obligation requires admissibility at every active state admitted by these control transitions; the delay counts control steps rather than elapsed time.

Algorithm 1 · Airlock core with parallel proof checking.   
Require: Contract K, certificate $\pi _ { \mathcal { K } } ,$ checker   
1 $\nu \gets [ ] ; k \gets 0 ; k _ { \mathsf { l a s t } } \gets - 1$   
2 if not contract\_check $( \pi \kappa , \mathcal { K } )$ or is\_active() then reject   
3 $\mathsf { e x e c } _ { \mathsf { c t r } \mathsf { l } } ( )$ ⊲ initialization   
4 while is\_active()   
5 �[�] ← read\_view(); publish\_view(�, �[�])   
6 if tool\_call\_ready()   
7 $( \mathsf { T } , i , k _ { 0 } , n , \pi ) \gets \mathsf { g e t } _ { - }$ \_tool\_request()   
8 $\mathsf { s t a r t \_ p r o o f \_ c h e c k } ( \mathsf { T } , i , k _ { 0 } , n , \pi , k _ { \mathsf { l a s t } } , \nu )$ ⊲ Check � against $\mathrm { O b l } _ { \top } ^ { \leqslant n } ( i , \upsilon [ k _ { 0 } ] )$   
9 if is\_proof\_correct() and $k \leq k _ { 0 } + n$   
10 exec<sub>T</sub> (�); $k _ { \mathrm { l a s t } }  k ;$ consume\_proof()   
11 if is\_active()   
12 $\mathsf { e x e c } _ { \mathsf { c t r } \mathsf { l } } ( ) ; k \gets k + 1$

Definition 4 Delayed obligation

For $s , s ^ { \prime } \in \Sigma _ { \mathrm { a c t } } ,$ set $\mathsf { D } _ { \mathsf { c t r l } } ^ { \leqslant 0 } ( s , s ^ { \prime } ) : = ( s = s ^ { \prime } )$ and define recursively, for $n \geq 0 _ { : }$

$$
\mathsf { D } _ { \mathsf { c t r l } } ^ { \leqslant n + 1 } ( s , s ^ { \prime } ) : = \mathsf { D } _ { \mathsf { c t r l } } ^ { \leqslant n } ( s , s ^ { \prime } ) \vee \exists u \in \Sigma _ { \mathsf { a c t } } , \quad \mathsf { D } _ { \mathsf { c t r l } } ^ { \leqslant n } ( s , u ) \wedge \mathsf { D } _ { \mathsf { c t r l } } ( u , s ^ { \prime } ) .\tag{5}
$$

The first disjunct keeps the paths already admitted, and the second extends them by one control step. Thus $\mathsf { D } _ { \mathsf { c t r } \mathsf { l } } ^ { \leqslant n } ( s , s ^ { \prime } )$ describes the active states $s ^ { \prime }$ admitted from � by at most � control steps through active states. For a tool T, input $i \in I _ { \mathsf { T } }$ , and view $\nu ,$ the delayed obligation is

$$
\begin{array} { r l } { \Xi } & { \mathrm { O b l } _ { \mathsf { T } } ^ { \leqslant n } ( i , v ) : = \forall s , s ^ { \prime } \in \Sigma _ { \mathrm { a c t } } , \mathsf { \Gamma } \bigl ( \mathsf { V i e w } ( s ) = \nu \wedge \mathsf { C } ( s ) \wedge \mathsf { D } _ { \mathsf { c t r l } } ^ { \leqslant n } ( s , s ^ { \prime } ) \bigr ) \Rightarrow \mathcal { H } _ { \mathsf { T } } ( i , s ^ { \prime } ) . } \end{array}\tag{6}
$$

The call must therefore be admissible at every state in the control window of every compliant active state compatible with the view.

For $n = 0$ , this recovers the ordinary obligation: $\mathrm { O b l } _ { \mathsf { T } } ^ { \leqslant 0 } ( i , \upsilon ) = \mathrm { O b l } _ { \mathsf { T } } ( i , \upsilon )$ . The control model includes intermediate states as well as states at completion, so the recursion can impose admissibility at more states than actual invocation boundaries permit. This makes the condition conservative. The intervening control calls remain subject to their own obligation.

## 3.4 The Airlock Core

The preceding definitions describe the states over which a contract is expressed, the tools through which the agent acts on the system, and the constraints that its evolution must respect. The obligations specify what must be established before a requested tool call can be admitted. We now bring these components together in the Airlock proper: the system that reads the state and publishes views, checks the supplied proofs, and coordinates the execution of tool and control routines. The Airlock core execution loop is summarized in Alg. 1.

The Airlock acts on the system through two operations: the agent’s calls and the control step ctrl, which it runs by itself. Control also serves as an observer: its repeated invocations provide fresh readings while the agent prepares requests and their proofs are checked. The system may evolve between readings. The Airlock keeps a record of its latest reading and a log: an append-only list of the readings and accepted requests with their proofs, which both parties can read.

![](images/734f0270fa3d1c46aabe6df1953d4465bb0f4b7823968faf92721800305eb75e.jpg)  
Figure 3. Execution with parallel proof checking. Above, a request based on $\nu = \nu [ k _ { 0 } ]$ is prepared and checked while the Airlock executes control. Dashed connectors trace receipt by the Airlock, proof checking and execution. The admitted call starts after six control invocations, within its requested window �. Control resumes after the call returns; the following view is published after that control invocation. Below, the dark dashed curve shows one quantity of the system state, and the coral step line the record: what the Airlock last read, at the end of each operation. Gray and blue regions show the bounds admitted by the control and tool models during their respective invocations; the pale background represents the contract’s constraint on this quantity.

Delayed calls. Preparing a request and checking its proof take time, during which the state may change. The Airlock therefore uses the delayed obligation $\mathrm { O b l } _ { \mathsf { T } } ^ { \leqslant n } ( i , \upsilon )$ of Definition 4: the proof must establish admissibility after up to � intervening control steps, counted from the view on which the request is based. Control continues while the request is prepared and checked. The current request can be executed once its proof is accepted, provided the session is still active and the allowance has not been exceeded. Fig. 3 illustrates this sequence; Sec. 4 states the conditions under which it preserves compliance.

## 3.4.1 The Algorithm

The Airlock invokes agent tools and control sequentially: after invoking a routine, the execution loop resumes only when that invocation returns normally. Proof checking runs in parallel, allowing the Airlock to continue invoking ctrl while a request is prepared and checked. The system may change both during execution and between observations.

Each operation ends with a reading obtained by accessing the system; this reading becomes the record and is appended to the log. These end-of-operation readings are implicit in Alg. 1. Initially, the record is a reading of the state taken when the Airlock starts. The routine read\_view() returns View(�) for the recorded state �, and is\_active() evaluates not stop(�) on that record.

Initialization. The guarantees hold from the moment the contract starts, so the contract must start in a state that satisfies its clauses. The Airlock cannot assume such a state, since nobody has observed the system at signing, and it cannot check one, since C need not be decidable. The control step establishes it, in its role as initializer. The loop receives K and a certificate �<sub>K</sub> with the evidence for all the assumptions on which the guarantees rest, as described in Sec. 6. The Airlock checks it with contract\_check(� , K) and reads the state.

(1) If the certificate is rejected, or the state is already active, the Airlock rejects the contract:

it never takes over a running contract, whose compliance it could not vouch for.

(2) Otherwise it runs one control step, in which ctrl acts as initializer: it either moves the system into an active state, which satisfies $\mathsf { C } ,$ as the certificate proves, or leaves it inactive.

(3) In the first case, we say that the contract starts, and the main loop begins. In the second, the contract never starts, and no call is ever made.

The agent plays no part in initialization: no view is published before the contract starts. The main loop exposes views and receives requests accompanied by proofs. Here, we describe these requests as a general interface, independently of how they are produced. However, in practice these are intended to be supplied by an agent.

Checking proofs. The initial contract check and the subsequent checks performed at runtime rely on mechanized reasoning, which allows mathematical statements and arguments to be expressed in a form that a computer can check (de Moura and Ullrich, 2021). The checker verifies that the supplied proof establishes the required statement. We leave the representation and construction of proofs implicit and describe which statements must be proved and when. Further details are given in Appendix B.

Main loop. After initialization, each cycle of Alg. 1 begins by checking whether the state is active. If so, the Airlock obtains �[�] ← read\_view() and publishes $( k , \upsilon [ k ] )$ through its output interface, making the view and its index available to the agent. Completed control steps provide the loop’s clock: the index � counts these steps rather than elapsed time, $\upsilon [ 0 ]$ is the view at the start of the contract, and subsequent views are obtained after the corresponding control invocations. Recorded views are kept unchanged so that a request can identify the view on which its proof is based.

The Airlock then checks for requests without blocking. A request $( \mathsf { T } , i , k _ { 0 } , n , \pi )$ identifies the call, the recorded view $\nu [ k _ { 0 } ]$ , the control-step allowance $n ,$ and a proposed proof $\pi$ of the delayed obligation $\mathrm { O b l } _ { \top } ^ { \leqslant n } ( i , \upsilon [ k _ { 0 } ] )$ from Definition 4. The allowance counts all control steps since that view, including those taken before the request arrives and during checking. The routine start\_proof\_check replaces any previous request and clears its verdict before starting the new check in parallel. It also checks that the tool and input are allowed and that the referenced view was published after the most recent agent call, so that only control steps intervene. The index $k _ { \mathrm { l a s t } }$ records that last call; its initial value −1 permits requests based on �[0]. These checks are detailed in Appendix B.

If the current request’s proof has been accepted and $k - k _ { 0 } \leq n ,$ the Airlock invokes exec (�) on the system as it then stands and waits for the routine to return. Agent tools are thus proof-gated. The delayed obligation requires admissibility at every active state admitted by the modeled control window. After the call, the Airlock records $k _ { | \mathrm { a s t } } = k$ and clears the accepted verdict through consume\_proof(), so that the request authorizes only one invocation.

Finally, if the state is still active, the Airlock invokes $\mathsf { e x e c } _ { \mathsf { c t r } \mathsf { l } } ( )$ and increments � when control returns. Thus control runs in cycles with no admitted call and also follows an admitted call unless that call ends the session. The next cycle checks whether the state is still active before obtaining and publishing a new view. The upper part of Fig. 3 illustrates this sequence and the control steps counted by the allowance �.

## 3.5 Tool design

The following examples illustrate how execution routines, transition models, and obligations fit together. We first give a possible design for the Airlock’s control tool, then consider families of agent tools and the constraints they can enforce.

![](images/fe6e569b5627d011b813152a24945ffcc2bcc9cf3b41b45bbe95812d1ce0d0fb.jpg)  
Figure 4. How ctrl can keep the system in homeostasis. (a) In the illustrated case $A c t = C \backslash C _ { \mathrm { i d l e } } ,$ , control idles in $C _ { \mathrm { i d l e } } ;$ in Act ∩ Fix (orange hatching) it fixes the state, moving it well inside $C _ { \mathrm { i d l e } }$ (small dashed ellipse); in End (red hatching) it terminates the contract. The darker area inside the dotted ellipse shows the reach of one idle period from $C _ { \mathrm { i d l e } }$ . (b) A call based on the view of state � may be invoked after up to � control steps. The blue-gray dotted region, $\mathsf { D } _ { \mathsf { c t r } \mathsf { l } } ^ { \leq n } ( s , \cdot )$ , represents active states admitted by the control models as possible invocation states; the delayed obligation requires the call to be admissible at each of them.

## 3.5.1 The control tool ctrl

Consider a control routine that plays the two roles of Definition 2.

As initializer, it tests whether the system is in a decidable region Ready of inactive states: those in which everything the contract needs at its start is in place, for instance the escrow in custody and the deadline ahead. There it runs $\mathsf { e x e c i n i t }$ , which initializes the contract: it moves the system into an active state that satisfies �, for instance by recording the start time and setting a status to ongoing. From the other inactive states the routine leaves the contract inactive; a contract may record that it did not start, as Example 1 does with the label void.

As controller, it keeps the active contract in $C ,$ as illustrated in Fig. 4(a). Fix an idle period $p$ and let $C _ { \mathrm { i d l e } }$ be the set of states � from which idling for one period remains in $C \colon C ( s ^ { \prime } )$ holds for every $s ^ { \prime }$ with $\mathsf { D } _ { \mathsf { i d l e } } ( p , s , s ^ { \prime } )$ . Like $C ,$ the set $C _ { \mathrm { i d l e } }$ need not be decidable, so the routine tests membership in a decidable region $A c t \supseteq C \setminus C _ { \mathrm { i d l e } }$ , which contains every compliant state from which idling may not be safe. In $A c t$ , the control step cannot idle until the next operation, since the contract could break in the meantime: it must act, and by default it terminates the contract.

Optionally, termination can be avoided by a fixing routine $\mathsf { e x e c } _ { \mathsf { f i x } }$ that, starting from a compliant state in a decidable region Fix, remains in � and, when it completes, leaves the system in $C _ { \mathrm { i d l e } }$ The routine then terminates only in End := Act \ Fix, where it must act but cannot fix.

The routine reads the state when it starts, its decision reading, and tests on it stop, Ready, Act and $F i x ,$ all decidable:

$$
{ \begin{array} { r l r l } & { { \mathrm { r o u t i n e ~ e x e c _ { c t r 1 } ( \lambda ) : } } } \\ & { \quad _ { s } \gets { \mathrm { r e a d } } \_ { - } { \mathrm { s t a t e ( \lambda ) } } } \\ & { \quad _ { { \mathrm { ~ i f ~ } } s \in \mathsf { D } ( s ) } \mathrm { t h e n } } & & { \mathrm { ~ s ~ } i n i t i a l i z e r } \\ & { \quad _ { \quad } \quad _ { { \mathrm { ~ i f ~ } } s \in R e a d y ~ \mathrm { t h e n ~ e x e c _ { \mathrm { i n i t } } ( \lambda ) } } } \\ & { \quad _ { { \mathrm { ~ e l s e ~ i f ~ } } s \notin A c t ~ \mathrm { t h e n ~ e x e c _ { \mathrm { i d l e } } ( p ) } } } & & { \mathrm { ~ s ~ } c o n t r o l l e r } \\ & { \quad _ { { \mathrm { ~ e l s e ~ i f ~ } } s \in F i x ~ \mathrm { t h e n ~ e x e c _ { \mathrm { f i x } } ( \lambda ) } } } \\ & { \quad _ { { \mathrm { ~ e l s e ~ e x e c _ { \mathrm { t e r m i n a t e } } ( \lambda ) } } } . } \end{array} }
$$

Here read\_state() accesses the system to obtain a state reading. The model $\mathsf { D } _ { \mathsf { c t r } 1 }$ must cover the states encountered while reading and carrying out the selected action, including changes between the two. These choices make both roles safe, as the certificate must show (Sec. 4.1). As initializer, the routine makes the contract active only through $\mathsf { e x e c i n i t : }$ , which ends in $C ;$ elsewhere the contract stays inactive, provided that stop depends only on a status that the routines set, such as the label of Example 1. As controller, it idles only in $C _ { \mathrm { i d l e } } .$ , fixes only where fixing keeps �, and otherwise terminates, which must also keep � until the contract closes. Fig. 4(a) illustrates the controller for $A c t = C \setminus C { \mathrm { i d l e } }$ . Panel (b) shows how it relates to delayed obligations: a call based on an earlier view must remain admissible throughout the region admitted by up to � control steps. Further design choices and applications are discussed in Appendix C.

## 3.5.2 Agent tools

Preconditions. Given any tool T, and a predicate Pre over $I \tau \times \Sigma$ we can define a new tool $\mathsf { T } | _ { \mathsf { P r e } }$ with same input $I _ { \mathsf { T } | _ { \mathsf { P r e } } } = I _ { \mathsf { T } }$ and same execution exec $_ { \cdot \mathsf { T } | _ { \mathsf { P r e } } } = { \mathsf { e x e c } } _ { \mathsf { T } }$ but with restricted transition relation

$$
D _ { \mathsf { T } | \mathsf { P r e } } ( i , s , s ^ { \prime } ) \quad : = \quad \mathsf { P r e } ( i , s ) \wedge D _ { \mathsf { T } } ( i , s , s ^ { \prime } ) .\tag{7}
$$

The resulting efect of this restriction is that enabledness becomes equivalent to $\Xi _ { \mathsf { T } | _ { \mathsf { P r e } } } ( i , s ) \iff$ $\mathsf { P r e } ( i , s ) \wedge \Xi _ { \mathsf { T } } ( i , s )$ and the corresponding obligation is

$$
\mathrm { O b l } _ { \mathsf { T } | _ { \mathsf { P r e } } } ( i , \upsilon ) \quad \Longleftrightarrow \quad \forall s \in \Sigma _ { \mathrm { a c t } } \quad \bigl ( \mathsf { V i e w } ( s ) = \upsilon \wedge \mathsf { C } ( s ) \bigr ) \quad \Longrightarrow \quad \bigl ( \mathsf { P r e } ( i , s ) \wedge \mathcal { A } _ { \mathsf { T } } ( i , s ) \bigr ) ,\tag{8}
$$

efectively adding Pre as a precondition gate that needs to be satisfied before allowing the original exec<sub>T</sub> to be called. Thus authorization requires proving both the gate and preservation of the contract’s constraints at every compliant active state compatible with the view. The gate constrains the starting state $s ,$ while $\Theta _ { \mathsf { T } }$ requires compliance at every admitted $s ^ { \prime } ,$ including during execution. At states satisfying Pre, the original transition relation is unchanged, so any overapproximation property is preserved there. Below we consider an example where adding a precondition to a tool is particularly useful.

Settlement tools. As illustrated in Example 1, the settle tools provide the main avenue for a contract to specify 1) the task for which the Airlock is instantiated and 2) what should happen upon its completion. These tools combine a goal predicate Goal over Σ with an execution routine $\mathsf { e x e c } _ { \mathsf { s e t t l e } }$ that performs the agreed settlement, such as releasing an escrow or delivering an artifact. Start with a tool free\_settle that has no caller-supplied arguments and whose execution routine performs this settlement, then impose Goal as its precondition: settle = free\_settle|<sub>Goal</sub>. Its model includes states during settlement, such as a payment in progress, as well as completed or failed outcomes. Suppose the contract certificate establishes that free\_settle is admissible at every compliant active state where Goal holds. The obligation then reduces to

$$
\mathrm { O b l } _ { \mathsf { s e t t l e } } ( \nu ) \iff \forall s \in \Sigma _ { \mathrm { a c t } } . \left( C ( s ) \wedge \mathsf { V i e w } ( s ) = \nu \right) \implies G o a l ( s ) ,\tag{9}
$$

namely that settle’s obligation holds exactly when every compliant active state compatible with the view satisfies the goal. In particular, when goal achievement can be determined by $\upsilon = \mathsf { V i e w } ( s )$ alone, we have $\mathrm { O b l } _ { \mathsf { s e t t l e } } ( \mathsf { V i e w } ( s ) ) \Longleftrightarrow G o a l ( s )$ for any active state � satisfying $C ( s )$ . For a delayed call, the goal must hold at every active state admitted by the control window. Preservation during settlement must already have been established in the certificate: for example, the clauses must allow a payment in progress, accounting for funds in transit, rather than only its final settled state.

Expert tools. Some clauses in the contract depend on judgments that the parties cannot readily express as a formal predicate: whether a document meets a requirement, an image shows a defect, or a hotel is suitable. The parties can instead agree on an expert to make these assessments. The tool’s routine exe $\mathsf { \widehat { \Pi } } _ { \mathsf { \mathtt { e } } \times \mathsf { p } \in \mathsf { r } \mathtt { t } }$ invokes the “expert”: this can be an LLM with a fixed prompt specified by the contract to extract information from a document or assess its contents, and records the answer. A vision model or human reviewer can play the same role. The model $\mathsf { D } _ { \mathsf { T } }$ covers the states while the assessment is pending and the possible recorded answers, as well as costs or disclosures incurred by consulting the expert. For example, a contract might reserve parts of the state to record the results of an expert’s calls to be used later on during task execution. In the travel contract, a booking requires a recorded hotel rating of at least 7; a pending assessment provides no such evidence. Recorded verdicts can be reused via memoization for the same object and evidence to avoid repeated sampling. Expert tools thus allow contracts to combine formal conditions with assessments of unstructured information and qualitative requirements.

Communication tools. E-mails, document uploads, and messages between agents add to a history of exchanges. A contract may constrain the recipient, the content disclosed, or the conditions under which a message may be sent. As a simple example, consider a tool send with input space $I _ { \mathsf { s e n d } } = D o c \times R e c$ , whose routine $\mathtt { e x e c } _ { \mathtt { s e n d } } ( d , r )$ sends document � to recipient �. Let Sent(�) record the document–recipient pairs disclosed so far, including partial disclosures. If no other disclosures occur during the call, a simple model is

$$
\begin{array} { r l r } { { \sf D } _ { \sf s e n d } ( ( d , r ) , s , s ^ { \prime } ) } & { { } : = } & { S e n t ( s ) \subseteq S e n t ( s ^ { \prime } ) \subseteq S e n t ( s ) \cup \{ ( d , r ) \} . } \end{array}\tag{10}
$$

This admits both states before disclosure and states in which some or all of � has been disclosed to $r ,$ whether or not delivery eventually succeeds. Let the contract constrain disclosure by $\mathsf { C } ( s ) : = S e n t ( s ) \subseteq A l l o w e d .$ . Assuming the state space admits the updated record for every state and input, send is enabled and, at a compliant active state �, its obligation reduces to

$$
\mathrm { O b l } _ { \mathsf { s e n d } } ( ( d , r ) , \mathsf { V i e w } ( s ) ) \quad \iff \quad ( d , r ) \in A l l o w e d .\tag{11}
$$

More detailed models can distinguish which contents are disclosed and through which routes;   
a failed delivery does not undo information already disclosed.

Query tools. A query tool performs operations such as web searches, database queries, and measurements. Its routine $\mathsf { e x e c } _ { \mathsf { q u e r y } }$ issues a query � and records the response with its source and, where relevant, time or version. Information exposed through the resulting view can support subsequent proofs: in the travel contract, a booking confirmation from the agreed service may provide evidence for goal achievement. Starting from a general tool query, each contract can impose its own restrictions through $\mathsf { q u e r y } | _ { \mathsf { P r e } } ,$ , retaining the same input space and execution routine. For example, a contract may require queries not to modify the queried data and to respect a spending limit �:

$$
\mathsf { P r e } ( q , s ) : = \mathsf { R e a d O n l y } ( q ) \wedge \big ( \mathsf { s p e n t } ( s ) + \kappa ( q ) \leq B \big ) .\tag{12}
$$

Here ReadOnly(�) expresses the first requirement and spent(�) is cumulative expenditure. The bound �(�) applies throughout execution: every state $s ^ { \prime }$ admitted by $\mathsf { D } _ { \mathsf { q u e r y } } ( q , s , s ^ { \prime } )$ satisfies spent $( s ^ { \prime } ) \leq { \mathsf { s p e n t } } ( s ) + \kappa ( q )$ , including while waiting for a response or after failure. By (8), the proof must establish this gate together with admissibility at every compliant active state compatible with the view.

The model must include states before a response is available, partial responses, errors, and timeouts as well as successful completion. Enabledness only requires that some state is admitted; it guarantees neither completion nor a successful answer. In particular, a timeout does not establish that no matching record exists. Records support only the claims justified by the model: a response may describe a particular database version, returned text need not be true, and reported availability is not a reservation. Read-only queries may still incur costs or disclose information, and the queried data may change independently during the call; these changes must also be modeled.

## 4 Guarantees

Sec. 3 presented the structure of PC2 and the obligations governing tool execution. We now study the guarantees that follow from this structure. We first show that the state satisfies C at every instant, regardless of the requests and proposed proofs supplied to the Airlock (Sec. 4.1). Sec. 5 relates the state to the world it describes and identifies conditions under which compliance implies safety.

## 4.1 Anytime compliance to the contract

Given a contract $\mathcal { K } : = ( \Sigma , \mathsf { C } , \mathsf { V i e w } , \mathcal { T } , \mathsf { c t r l } , \mathsf { s t o p } )$ , we identify suficient conditions for the state to satisfy C at every instant of the execution of Alg. 1, for arbitrary incoming requests and proposed proofs.

For a given execution, we write $s ( \tau )$ for the possibly unobserved state of the system at time �, in the sense of Sec. 3. As described in Sec. 3.4, the Airlock invokes control and tool routines sequentially. Each execution interval starts when a routine is invoked and ends when it returns, or continues indefinitely if it does not return. During this interval the state may change through the routine’s actions and outside changes (see Fig. 3).

We begin with the proof checks introduced in Sec. 3.4. Before executing a requested tool call, the Airlock forms its obligation from the contract and the published view, and passes it, together with the supplied proof, to the checker: a program that checks whether the supplied argument establishes the required statement according to the rules of a fixed logic. For our implementation, we choose dependent type theory as the underlying logic: it provides a common language for the contract’s data, statements, and proofs, with proof checking reduced to type checking. A statement is represented by a type, and the checker verifies that the supplied proof is an expression of that type. In Lean, used for our mechanization (Sec. 7), this check is performed by the Lean kernel (de Moura and Ullrich, 2021).

## A1 Sound checking

For any proof � and statement Φ, if the proof-checker accepts � as a proof of Φ, then Φ holds under the axioms and interpretation of the chosen logic.

A1 is independent of the choice of proof assistant. The checker must use the contract’s agreed definitions and permitted axioms. Accepting a proof of $\mathrm { O b l } _ { \mathsf { T } } ^ { \leqslant n } ( i , \upsilon )$ then establishes that obligation, allowing the analysis below to rely on it regardless of how the proof was produced. The obligation, however, is expressed in terms of the transition models $\mathsf { D } _ { \mathsf { T } }$ and $\mathsf { D } _ { \mathsf { c t r } 1 }$ , while the state evolves through the execution routines and the outside changes. To apply the proof to execution, the models must cover every state that execution can reach. We denote these possibilities by the efect relation $E _ { \mathsf { T } }$ on $I _ { \mathsf { T } } \times \Sigma \times \Sigma \colon E _ { \mathsf { T } } ( i , s , s ^ { \prime } )$ holds if and only if an invocation of $\mathsf { e x e c } _ { \mathsf { T } } ( i )$ starting in � can reach $s ^ { \prime }$ during its execution. This includes initial, intermediate and final states, outside changes during the invocation, and states reached if the routine never returns.

## A2 Coverage of state transitions

For every ${ \mathsf { T } } \in { \mathcal { T } } \cup \{ { \mathsf { c t r } } { \mathsf { l } } \}$ ,

$$
\forall i \in I _ { \mathsf { T } } , \ s , s ^ { \prime } \in \Sigma \qquad \Xi _ { \mathsf { T } } ( i , s ) \wedge E _ { \mathsf { T } } ( i , s , s ^ { \prime } ) \implies \mathsf { D } _ { \mathsf { T } } ( i , s , s ^ { \prime } ) .\tag{13}
$$

A2 requires $\mathsf { D } _ { \mathsf { T } }$ to over-approximate these possibilities wherever $\Xi _ { \mathsf { T } } ( i , s )$ holds: it may admit states through which the state never passes, but must include every state through which it may pass during the operation, not only the one at its end. The restriction to enabled calls due $\mathrm { t o } \ E \tau$ allows $\mathsf { D } _ { \mathsf { T } }$ to exclude inputs and starting states, as in the precondition construction of Sec. 3.5.2; the obligation then prevents authorization of those calls. The Airlock uses $\mathsf { D } _ { \mathsf { T } }$ to form obligations and does not need to know $E _ { \mathsf { T } }$ . The role of $E _ { \mathsf { T } }$ in the analysis is to specify the behavior that the model must cover for these proofs to apply to execution. Both relations concern the state; Sec. 5 and 6 discuss how a deployment’s routines, services and host support this coverage and the state’s correspondence with the world.

The obligation preserves compliance from a state that already satisfies C. To apply it throughout the loop, the loop must start from a compliant state, and the control step, which the Airlock runs by itself, must preserve compliance too. Nobody has observed the system at signing, so its initial state is not known to comply. The Airlock does not assume it: the contract begins inactive, and the control step, as initializer, makes it active only in a compliant state.

## A3 Initialization and control

The control tool satisfies

$$
\begin{array} { r l r l r l } { \forall s \in \Sigma _ { \mathrm { a c t } } } & { } & { \mathsf { C } ( s ) \implies \mathcal { A } _ { \mathrm { c t r l } } ( s ) , } & { \qquad } & { } & { ( c o n t r o l l e r ) } \\ { \forall s \in \Sigma _ { \mathrm { s t o p } } , s ^ { \prime } \in \Sigma _ { \mathrm { a c t } } } & { } & { \mathsf { D } _ { \mathrm { c t r l } } ( s , s ^ { \prime } ) \implies \mathsf { C } ( s ^ { \prime } ) , } & { } & { } & { ( i n i t i a l i z e r ) } \end{array}
$$

and it is enabled at every inactive state: $\Xi _ { \mathsf { c t r l } } ( s )$ for every $s \in \Sigma _ { \mathsf { s t o p } }$

The two lines are the two roles of ctrl. As controller, from a compliant active state, it passes only through compliant states. As initializer, from an inactive state, it can make the contract active only in a compliant state. Both are certified once, at startup (Sec. 3.4), so no new proof is required at each cycle, and the second replaces the assumption that the initial state complies, which nobody could check. Enabledness makes A2 applicable to every control step as well as to calls.

Finally, the argument relies on the Airlock running Alg. 1 as specified.

## A4 Execution integrity

Execution follows Alg. 1 and the proof-checking routine in Appendix B. In particular:

(i) execution is a sequence of contiguous operations, each a tool call or a control step: the first starts when the Airlock has read the state, and each of the others when the previous one ends;

(ii) after each routine returns, the current state is read and appended to the log; views and the inactivity test are computed from this reading, and initially from a reading taken when the Airlock starts;

(iii) a call runs only for the pending request $( \mathsf { T } , i , k _ { 0 } , n , \pi )$ , at most once, if no other call ran since $\nu [ k _ { 0 } ]$ was published, in an iteration � with $k _ { 0 } \leq k \leq k _ { 0 } + n ,$ and after the proof-checking routine in Appendix B has checked that ${ \mathsf { T } } \in { \mathcal { T } } ,$ , that $i \in I _ { \mathsf { T } } ,$ and that � proves $\mathrm { O b l } _ { \mathsf { T } } ^ { \leqslant n } ( i , \upsilon [ k _ { 0 } ] )$ , a statement the Airlock forms itself;

A4(i) lets us follow the operations one after another: the state at the end of one is also the state at the start of the next. In practice, between two operations the Airlock reads the state, publishes a view and invokes the next routine, which takes microseconds, while requests take seconds to prepare. Proof checking runs in parallel with control, so the state need not remain fixed while a proof is checked. Appendix D discusses executions with bounded gaps between routines.

A4(ii) connects execution to the information used by the loop. After each routine, the Airlock records the current state. Whenever it publishes a view or tests whether the session is still active, it uses the latest reading. Thus these decisions reflect the result of the routine that just

ran.

Together, A4(i–ii) ensure that the latest recorded state is the state from which the next routine starts. Part (iii) then connects that invocation to its authorization: the checked proof concerns the requested tool and input, and the call runs only within its control window, with no intervening tool call or reuse of the request.

A request may refer to a view published several control steps before its call starts. To apply its delayed obligation, we must show that the actual starting state is among those covered by that obligation. The following lemma connects the request’s counter window to the states admitted by $\mathsf { D } _ { \mathsf { c t r } 1 } ^ { \leqslant n }$

## Lemma 1 Window

Assume A2 – A4 . Let a call run for the request $( \mathsf { T } , i , k _ { 0 } , n , \pi )$ from state $s ^ { \prime } ,$ and let $\bar { s }$ be the statefrom which $\nu [ k _ { 0 } ]$ was computed. $I f C$ holds at every instantfrom the start of the contract to the invocation, then ¯� and $s ^ { \prime }$ are active and $\mathsf { D } _ { \mathsf { c t r } \mathsf { l } } ^ { \leqslant n } ( \bar { s } , s ^ { \prime } )$

Proof. Let � be the iteration in which the call runs. By A4(iii) , $k _ { 0 } \leq k \leq k _ { 0 } + n ,$ and only control invocations intervene between ¯� and the call’s starting state $s ^ { \prime } .$ . There are $m = k - k _ { 0 } \leq n$ such invocations. By A4(i–ii) , they are contiguous, and their final states are the recorded readings. For $m = 0 , \bar { s } = s ^ { \prime }$ . The loop computes a view and starts an iteration only from a reading that it has found active ( A4(ii) ). So ¯�, the starting state of each of these control steps, and $s ^ { \prime }$ are active. Each control step started from a compliant state by hypothesis, since it runs after the contract has started. Hence A3 gives its enabledness, and A2 places its final state in $\mathsf { D } _ { \mathsf { c t r l } }$ from its starting state. Chaining these steps through active states gives $\mathsf { D } _ { \mathsf { c t r } \mathsf { l } } ^ { \leqslant m } ( \bar { s } , s ^ { \prime } )$ , hence $\mathsf { D } _ { \mathsf { c t r } \mathsf { l } } ^ { \leqslant n } ( \bar { s } , s ^ { \prime } )$ , since $m \leq n$ and each window contains the previous one (Definition 4). ■

The lemma explains why the state need not remain unchanged after a view is published: given compliance up to the call, its delayed obligation covers the actual starting state despite the intervening control steps. The window bounds the number of those steps, not the elapsed time.

## Theorem 1 Anytime compliance to the contract

Under A1 – A4 , for any sequence of tool-call requests and proposed proofs supplied to $A l g . \ 1 ,$ every resulting execution in which the contract starts satisfies C at every instant from the start until the contract closes, closing included, or at every instant after the start if it never closes.

Proof. By A4(i) , operations are contiguous. Let $s _ { 0 }$ be the state when the Airlock starts. Initialization runs a control step from $s _ { 0 }$ , which is inactive, since otherwise the Airlock rejects the contract ( A4 ). By A3 the control step is enabled at $s _ { 0 } ,$ so A2 places the state $s _ { 1 }$ at its end in $\mathsf { D } _ { \mathsf { c t r } ^ { 1 } } ( s _ { 0 } , \cdot )$ . Since the contract starts, �<sub>1</sub> is active, and A3 gives ${ \sf C } ( s _ { 1 } )$ . It therefore sufices to prove, by induction on the invocations after the start, that each starts in a compliant state and preserves compliance throughout its execution. The first starts from $s _ { 1 }$ . Consider an operation that starts from the state $s ,$ with C(�) by induction; � is active, since Alg. 1 runs operations only from such states.

(1) A control step. A3 gives $\mathcal { A } _ { \sf c t r 1 } ( s )$ , that is $\Xi _ { \mathsf { c t r l } } ( s )$ and $\Theta _ { \mathsf { c t r l } } ( s )$ . Every state $s ^ { \prime }$ reached during its interval satisfies $E _ { \mathsf { c t r l } } ( s , s ^ { \prime } )$ , so A2 gives $\mathsf { D } _ { \mathsf { c t r } ^ { 1 } } ( s , s ^ { \prime } )$ , hence ${ \mathsf { C } } ( s ^ { \prime } )$

(2) A call T(�), for the request $( \mathsf { T } , i , k _ { 0 } , n , \pi )$ . By A4(iii) , the checker accepted $\pi$ as a proof of $\mathrm { O b l } _ { \mathsf { T } } ^ { \leqslant n } ( i , \upsilon [ k _ { 0 } ] )$ , which therefore holds by A1 . Let ¯� be the state from which � $[ k _ { 0 } ]$ was computed: $\mathsf { V i e w } ( \bar { s } ) = \nu [ k _ { 0 } ]$ , and ${ \mathsf { C } } ( { \bar { s } } )$ by induction. By Lemma 1, ¯� and � are active and $\mathsf { D } _ { \mathsf { c t r } 1 } ^ { \leqslant n } ( \bar { s } , s )$ . The obligation, applied to ¯� and $s ,$ gives $\mathcal { A } _ { \sf T } ( i , s )$ , that is $\Xi _ { \mathsf { T } } ( i , s )$ and $\Theta _ { \mathsf { T } } ( i , s )$ . Every state $s ^ { \prime }$ reached during the call’s interval satisfies $E _ { \mathsf { T } } ( i , s , s ^ { \prime } )$ , so A2 gives $\mathsf { D } _ { \mathsf { T } } ( i , s , s ^ { \prime } )$ , hence ${ \mathsf { C } } ( s ^ { \prime } )$

In both cases, compliance holds throughout the invocation. If it returns, its final state is compliant and, by A4(i) , it is the state from which the next operation, if any, starts. If it never returns, the same argument applies throughout its continuing execution. A rejected or expired request authorizes no call. The argument makes no assumption about how requests or proofs are produced.

Thm. 1 derives compliance of the whole execution from the individual obligations. What, then, does each obligation require? By Definition $4 , \mathrm { O b l } _ { \mathsf { T } } ^ { \leqslant n } ( i , \upsilon )$ holds exactly when the call is admissible from every active state admitted by at most � control steps, through active states, from a compliant active state with view �. It therefore accounts both for information hidden by the view and for changes during the control window. This condition is relative to the chosen models, which may admit states and paths that actual execution cannot follow. Establishing admissibility from all these states sufices for compliance, but a false obligation need not imply that the call would actually violate the contract. Relative to the models, nothing weaker would do: every rule that safely admits calls on the basis of the view implies the obligation, so the agent may follow any strategy whose calls keep the contract (Proposition A.1).

The theorem allows arbitrary requests and proposed proofs, but its coverage assumption still concerns every state that execution can reach, including through outside interactions. Sec. 5.2 examines how restricting the agent’s access can make this assumption easier to establish while retaining models that allow useful calls.

## 4.2 Completion through goal tools

A standard contract (Example 1) ends in one of three ways: it is void if the conditions to start do not hold, settled when the agent proves its goal, or terminated by the control step, for instance at its deadline with a refund. The following theorem shows that it ends in at most one of them, and in each only as agreed; Thm. 5 adds that, with a deadline, it does end. Besides A1 – A4 , the theorem uses that the settlement tool is settle = free\_settle| (Sec. 3.5.2), that the control step terminates the contract exactly when its decision reading is in the region End of Sec. 3.5.1, and the properties of the label stated in Example 1.

Theorem 2 Completion

Assume A1 – A4 . In every execution of Alg. 1 with a standard contract, the contract ends in at most one of the following ways:

(1) void: the conditions to start did not hold; the execution never starts and no tool-call is made;

(2) settled: the agreed settlement procedure ran, from a state where the goal held;

(3) terminated: the agreed termination procedure ran, after a control step found a termination condition.

Proof. The contract is created pending, and only the routines of ctrl and settle change the label, at their end (Example 1). The first operation is a control step, which, as initializer, sets the label to ongoing when its decision reading is in Ready, and to void otherwise; in the second case the loop stops, and no call is made. Once the label is settled or terminated, the contract is inactive and Alg. 1 runs no further operation, so the contract ends in at most one way. (2) As in case (2) of the proof of Thm. 1, the checker accepted a proof of $\left. \operatorname { O b l } _ { \mathsf { s e t t l e } } ^ { \leqslant n } ( \nu [ k _ { 0 } ] ) \right.$ , which therefore holds ( A1 ). Admissibility of settle includes its enabledness, hence Goal, by the precondition construction, at every active state of the window, and Lemma 1 places in the window the state from which the settlement started. (3) The label becomes terminated only when the control step has run the termination procedure, which it does exactly when its decision reading is in End. ■

The case analysis rests on the code of the Airlock and of the control step; case (2) also rests on the checker and on the obligation of settle.

At the call-selection step of an active cycle, a pending settlement request whose proof has been accepted and whose window remains valid is selected for execution without further authorization. The client therefore has no discretionary veto over an authorized settlement through the Airlock. The next proposition identifies exactly what its proof must establish.

Proposition 1 What settlement asks

Assume A3 for the standard contract, and let its certificate establish that free\_settle is admissible at every compliant active state where Goal holds. Then, for every view � and delay $n ,$

$$
\mathrm { O b l } _ { \mathsf { s e t t l e } } ^ { \leqslant n } ( v ) \iff \forall s , s ^ { \prime } \in \Sigma _ { \mathrm { a c t } } \quad \bigl ( \mathsf { V i e w } ( s ) = v \wedge \mathsf { C } ( s ) \wedge \mathsf { D } _ { \mathsf { c t r } 1 } ^ { \leqslant n } ( s , s ^ { \prime } ) \bigr ) \implies G o a l ( s ^ { \prime } ) .
$$

Proof. By the precondition construction, $\begin{array} { r } { \Xi _ { \mathsf { s e t t l e } } ( s ^ { \prime } ) \qquad \Longleftrightarrow \qquad G o a l ( s ^ { \prime } ) \wedge \Xi _ { \mathsf { f r e e \_ s e t t l e } } ( s ^ { \prime } ) } \end{array}$ and $\Theta _ { \mathrm { s e t t l e } } ( s ^ { \prime } ) \quad \Longleftrightarrow \quad \big ( G o a l ( s ^ { \prime } ) \quad \Longrightarrow \quad \Theta _ { \mathrm { f r e e \ s e t t l e } } ( s ^ { \prime } ) \big )$ . From left to right, $\mathcal { A } _ { \sf s e t t l e } ( s ^ { \prime } )$ gives $\Xi _ { \mathsf { s e t t l e } } ( s ^ { \prime } )$ hence Goal(�<sup>′</sup>). From right to left, ${ \mathsf { C } } ( s ^ { \prime } )$ holds, since $s ^ { \prime }$ is reached from the compliant state � by control steps from active states, which preserve C by A3 . Together with Goal(�<sup>′</sup>), the certificate gives $\mathcal { A } _ { \sf f r e e \_ s e t t l e } ( s ^ { \prime } )$ , hence $\mathcal { A } _ { \sf s e t t l e } ( s ^ { \prime } )$ ■

When every compliant active state with the given view satisfies the goal and modeled control steps preserve it throughout the window, the right-hand side holds. A proof of this condition sufices for authorization while the request remains valid; admissibility of the settlement procedure is already established by the contract certificate.

## 5 From compliance to safety

Sec. 4 establishes that execution remains within the contract’s constraints. We now study when this compliance protects the resources and interactions the contract concerns. A payment recorded within budget, for instance, does not establish that the service charged no additional fees. Connecting the two requires both a mathematical account of what the state describes and conditions on the system that produces and acts on that description. This is the gap between a specification, which a machine can be built to meet, and a requirement about the world in which the machine operates (Jackson, 1995; Zave and Jackson, 1997). A proof relates a system to a model of it, never the model to the world (Smith, 1985).

Three questions guide this section and the next:

• What does the mathematical construction guarantee? Under which assumptions does compliance of the state imply safety in the world?

• How can those assumptions be established for a deployment? What must the routines, host, access controls and external services do, and what evidence supports these claims?

• What do the parties do about what remains assumed? Which assumptions do they accept, who undertakes responsibility for them, and what happens if they fail?

Sec. 5.1 addresses the first question, and illustrates on an example what a world is. Sec. 5.2 studies how isolating the agent supports useful efect models and guarantees that apply to every agent under the same deployment conditions. Sec. 5.3 discusses the requirements on execution. Sec. 6 addresses the second and third questions. It shows that the assumptions of a deployment need not be established from scratch, since they follow by composing guarantees that ship with the capability modules from which its tools are built (Sec. 6.1). It then assembles these dependencies in a tree and records their proposed allocation to parties (Sec. 6.5), connecting this analysis to the treatment of remaining assumptions in Sec. 8.3.

## 5.1 World safety

Let $\mathcal { W }$ be a space of configurations of the relevant system and environment, including such facts as actual charges, issued payment instructions and the state of external services. We write $w ( \tau )$ for its configuration at time �. A relation $\rho \subseteq \mathcal W \times \Sigma$ specifies how the contract’s state describes these facts: $\rho ( w , s )$ means that � is a valid description of �. This description can be partial or conservative. A recorded upper bound on expenditure, for instance, can describe several diferent amounts actually charged, and the same expenditure can admit several valid bounds. In the terms of program refinement, $\rho$ is an abstraction relation between the world and its description (Hoare, 1972; Abadi and Lamport, 1991). It thus difers from View, which determines what the requesting agent sees of the state.

## Example 2 A trading agent and its world

Consider an agent that manages a small investment portfolio. Its tools read and send email, read and write files in a working directory, search, read and publish posts on a social network, read quotes from a market-data service, query a database ofpast transactions, and place orders through a brokerage account. A world � ∈ W is a configuration of all these systems: the messages in the mailbox and those delivered to their recipients, the files on disk, the posts that the public can read, the prices at which the securities currently trade, the rows of the database, and the cash, positions and pending orders held by the broker, together with the current time. The state � records only what the contract needs, as the tools report it: for instance, the messages sent with their recipients, the posts published, the last quote read with its time, and the money committed to orders. The correspondence � says what each record means in the world. The record of sent messages describes the world if every message that left the mailbox appears in it. A recorded quote describes the world if the service reported that price at that time, whatever the price is now. The recorded commitment may be a conservative bound, counting each pending order at its maximum cost. A protection such as “never commit more than a given amount, and never disclose the portfolio $p u b l i c l y '$ is a predicate Safe on worlds: it concerns the money actually committed and what the public can actually read, not the records. Much of the world changes without the agent: messages arrive, prices move, other applications update the database, and the broker executes pending orders.

In most scenarios, such as the example above, viewing a world $w \in \mathcal { W }$ as a single object is impractical. This is because applications are often organized in largely independent systems, so the resulting world can be naturally represented as a product of their configurations, ${ \mathcal { W } } = { \mathcal { W } } _ { \mathrm { m a i l } } \times { \mathcal { W } } _ { \mathrm { f i l e s } } \times \cdot \cdot \cdot \times { \mathcal { W } } _ { \mathrm { b r o k e r } }$ , and the state a product of the corresponding records. In Sec. 6 we will address this question and show how a world, a state, a correspondence and the tools that act on them can be assembled from small modules, each concerning a single service such as the mailbox or the brokerage account, whose guarantees are established once.

Let Safe be a predicate on W specifying the intended protection, as in Example 2 above. Worlds may include history when the protection concerns past events, such as a document having been disclosed, as history variables do in proofs of refinement (Abadi and Lamport, 1991). Compliance is expressed by C(�), while safety concerns Safe(�). We connect them through two conditions. Fidelity requires the state to remain a valid description of the world as execution proceeds. Adequacy requires a world described by a compliant state to satisfy the intended safety property. Together, they relate what the Airlock enforces on states to what the parties require of the world. This is the satisfaction argument of requirements engineering, in which a specification, together with properties of the world, entails a requirement (Zave and Jackson, 1997; Jackson, 2001).

Fidelity. To express preservation of correspondence, we must account for how the world and state evolve together. Let $\widehat { E } _ { \mathsf { T } } ( i , w , s , w ^ { \prime } , s ^ { \prime } )$ hold when an invocation of exec (�) starting in $( w , s )$ can reach $( w ^ { \prime } , s ^ { \prime } )$ at any instant of its execution. This includes outside changes during the invocation and applies also to a routine that does not return. The efect relation of Sec. 4.1 is

its projection on states:

$$
E _ { \mathsf { T } } ( i , s , s ^ { \prime } ) \qquad \Longleftrightarrow \qquad \exists w , w ^ { \prime } \in \mathcal { W } . \widehat { E } _ { \mathsf { T } } ( i , w , s , w ^ { \prime } , s ^ { \prime } ) .\tag{14}
$$

This joint relation lets us state what it means for execution to preserve the description: starting from a pair satisfying $\rho ,$ every pair reached during an enabled invocation must still satisfy it, as a simulation relation is preserved by every step of the system it relates (Lynch and Vaandrager, 1995). Correspondence must also be established when the contract starts, since nothing is known of the world when the contract is signed. We collect these requirements in the following assumption.

## A5 Fidelity

For every ${ \mathsf { T } } \in { \mathcal { T } } \cup \{ { \mathsf { c t r } } { \mathsf { l } } \}$ , input $i \in I _ { \mathsf { T } } .$ , worlds $w , w ^ { \prime }$ and states $s , s ^ { \prime } ,$

$$
\begin{array} { r l } { \rho ( w , s ) \wedge \Xi _ { \mathsf { T } } ( i , s ) \wedge \widehat { E } _ { \mathsf { T } } ( i , w , s , w ^ { \prime } , s ^ { \prime } ) } & { { } \implies \rho ( w ^ { \prime } , s ^ { \prime } ) , } \end{array}\tag{15}
$$

and, for every inactive state � and active state $s ^ { \prime } { } _ { ; }$ ,

$$
\begin{array} { r l } { \widehat { E } _ { \mathrm { c t r l } } ( w , s , w ^ { \prime } , s ^ { \prime } ) } & { { } \Longrightarrow \quad \rho ( w ^ { \prime } , s ^ { \prime } ) . } \end{array}\tag{16}
$$

A2 requires the model to cover the states reached during execution, while A5 requires those states to remain valid descriptions of the world. The claim that a service reported a confirmed booking at time � describes a past event. Once established, it survives a later cancellation if the record is preserved and the world model retains that history. A claim that the booking is currently confirmed must instead account for possible cancellations. Fidelity thus depends on what the recorded information means, while adequacy asks whether that meaning sufices for the intended protection. The second line is the counterpart of the initializer line of A3 : the control step makes the contract active only in a state that describes the world, whatever the world was before, so nothing needs to be assumed about the world when the contract is signed. For the control design of Sec. 3.5.1, a suficient condition is that read\_state() returns a state that describes the world when the reading completes, and that the remaining initialization preserves this correspondence until the contract becomes active.

Adequacy. Fidelity gives meaning to the state throughout execution. We must then ask whether the clauses imposed on that state sufice for the protection the parties require: even a faithful description can satisfy a contract that omits an important constraint. An agent that optimizes within the clauses pushes exactly where they and the intended protection come apart (Manheim and Garrabrant, 2018), and no contract specifies everything its parties care about (Hadfield-Menell and Hadfield, 2019).

## A6 Adequacy

For every $w \in \mathcal W$ and $s \in \Sigma ,$

$$
\rho ( w , s ) \wedge \mathsf C ( s ) \quad \Longrightarrow \quad \mathsf { S a f e } ( w ) ,\tag{17}
$$

and, for a contract with a goal Goal whose intended outcome in the world is a predicate $G o a l _ { \mathcal { W } } ,$

$$
\rho ( w , s ) \wedge G o a l ( s ) \quad \implies \quad G o a l _ { \mathcal { W } } ( w ) .\tag{18}
$$

For specified $\rho , { \mathsf { C } }$ and Safe, A6 can be proved when the relationships expressed by the correspondence and clauses imply the required protection. In the payment ledger of Appendix E, correspondence bounds actual expenditure by recorded reservations, and the clause bounds those reservations by the budget, so adequacy follows by arithmetic. The parties must also establish that Safe expresses their intended protection, since a proof of the implication alone does not settle this question.

We can now extend Thm. 1 from compliance of the state to safety in the world. The additional step is to show that the state is a valid description of the world throughout execution from the start of the contract, including while a routine runs.

## Theorem 3 World safety

Under A1 – A6 , every execution of Alg. 1 in which the contract starts satisfies $\mathsf { S a f e } ( \boldsymbol { w } ( \tau ) )$ at every instant from the start until the contract closes, closing included, or at every instant after the start if it never closes, for any sequence of requests and proposed proofs.

Proof. By Thm. 1, C(�(�)) holds from the start of the contract. When the control step starts the contract, the state describes the world, by the second line of A5 . Each later operation is enabled at its starting state: control by A3 , and a requested call by its accepted obligation and Lemma 1, as in the proof of Thm. 1. Hence the first line of A5 preserves correspondence throughout that operation, which, operations being contiguous ( A4(i) ), lasts until the next one starts, or throughout its continuing execution if it never returns. Induction over the operations therefore gives $\rho ( w ( \tau ) , s ( \tau ) )$ at every instant from the start. Together with compliance, A6 yields the conclusion. ■

Thm. 3 separates what must be checked for each call from what must be established about the deployment. The checker reasons about the contract’s states and tool models. Fidelity keeps these states valid descriptions throughout execution, and adequacy makes their compliance suficient for safety in the world. The argument transfers a property from the description to the world it describes, as a refinement mapping transfers a property proved of a specification to a system that implements it (Abadi and Lamport, 1991). This is what allows the local obligations to protect the resources and interactions the contract concerns.

Establishing these conditions requires examining which interactions the deployment permits, and Sec. 5.2 considers how the agent’s access afects them.

Many contracts also prescribe a settlement action once a goal has been achieved, as in the standard contract of Example 1. The completion guarantee of Thm. 2 concerns this goal as expressed on states. The following corollary transfers it to the intended outcome $G o a l _ { \mathcal { W } }$ in the world, through $\rho$ and the second part of A6 .

Corollary 4 Completion in the world   
Under the assumptions of Thm. 2, assume A5 and A6 . Then every invocation of the settlement routine starts at an instant � satisfying Goal<sub>W</sub> (�(�)).

Proof. As in the proof of Thm. 2, case (2), the accepted settlement obligation gives Goal(�(�)). By the correspondence argument in the proof of Thm. $3 , \rho ( w ( \tau ) , s ( \tau ) )$ holds, and (18) gives the conclusion. ■

A central question is to determine when assumptions A5 and A6 are satisfied. Appendix E works through step-by-step examples where they are derived in practice, for a payment ledger for safety and booking confirmations for completion, including settlement of the travel contract.

## 5.2 Complete mediation and sandboxing: safety against every agent

The agent’s access. The discussion so far has left one actor implicit: the agent, which may be adversarial. Suppose that the agent could act on the state or on the world independently of the Airlock, for instance by deleting or sending messages in a mailbox while a tool that reads the mailbox is running. Since $E _ { \mathsf { T } }$ and $\widehat { E } _ { \mathrm { T } }$ describe everything that can happen during an invocation of T, they would then include the agent’s direct actions, alongside the outside changes of the world. The possible efects can therefore depend on which agent is present. We write $E _ { \mathsf { T } } ^ { \alpha }$ and $\widehat { E } _ { \top } ^ { \alpha }$ for the efect relations of T when the agent is $\alpha .$ Coverage ( A2 ) requires each model $\mathsf { D } _ { \mathsf { T } }$ to cover these additional changes, and fidelity ( A5 ) requires the correspondence to survive them. One option is to model the particular agent, tying the justification of the tools to that agent’s behaviour. The other is to use a common model that covers every change any agent could cause. This avoids verifying the agent, but if those possibilities include a contract violation during every invocation, no call can be authorized, and control itself cannot satisfy A3 .

![](images/e3e8a849a9ca364deefcdf7e5223c41fad8acb4c474dee944eed2f78709145a5.jpg)  
Figure 5. What can go wrong without isolation. An action outside the proof gate (pale path) can make the world unsafe while control runs. Depending on how the state responds, one of the assumptions fails, as the three branches show (Appendix F).

An agent can also change the world without changing the state. If that change invalidates what the state describes, fidelity fails, and enlarging $\mathsf { D } _ { \mathsf { T } }$ alone cannot repair it. Fig. 5 illustrates these failures, and Appendix F works through them in detail. The classical answer to this problem is to restrict the agent’s access instead of modeling it. This is suficient, though, as we discuss at the end of this section, not necessary.

Controlling access in this way has a long history in computer security. In particular, the principle of complete mediation requires each access to the objects that a system protects to pass through the authorization that governs it (Saltzer and Schroeder, 1975). Here the protected objects are the state and the world, and the authorization is the proof gate: the Airlock checks an obligation before executing a requested operation ( A4(iii) ). Mediation is complete only if the agent has no other way to act, which is what isolation, stated below, provides. The Airlock then acts as a reference monitor for the agent’s access (Anderson, 1972).

To express what this restriction provides, we take the agent to comprise every process that runs for it: a model, the code that drives it, and any process they start. We then fix a tool T and its input $i ,$ and consider its routine running with that input while none of the agent’s processes runs. Other parties, external services and the passage of time remain present, under the same conditions on their behaviour. We call these reference executions. As with the efect relations of Sec. 4.1 and 5.1, $E _ { \intercal } ^ { \bullet } ( i , s , s ^ { \prime } )$ holds when such an invocation starting in � can reach $s ^ { \prime }$ at any instant, and $\widehat { E } _ { \top } ^ { \bullet } ( i , w , s , \dot { w } ^ { \prime } , s ^ { \prime } )$ records the corresponding possibility for a world–state pair. The following assumption specifies how the agent can be present without introducing efects beyond those of these executions.

## A7 Isolation of the agent

The agent acts on the system only through invocations selected by the Airlock among its requests, in particular:

(i) Nothing else it does, through any channel (e.g. files, network connections, shared memory or messages to other processes) changes the world, the state or the internal state of a running routine, during or between invocations of any ${ \mathsf { T } } \in { \mathcal { T } } \cup \{ { \mathsf { c t r l } } \}$ routine.

(ii) Each tool or ctrl routine receives from the agent only the call input, fixed at invocation.

(iii) The evolution of the world and the state during the invocation depends only on their current values, the routine’s internal state and the fixed input. In particular it does not depend on the agent’s computation or internal state.

The assumption concerns how the system is organized, rather than whether the agent behaves cooperatively. It allows the agent to choose requests and prepare proofs, but excludes both direct changes to the protected system and further influence on a running routine through shared data or pending requests. The independence condition lets us compare an execution with an agent to one of the reference executions. The prototypical way to establish A7 is to run the agent in a sandbox, a container or lightweight virtual machine (Goldberg et al., 1996; Agache et al., 2020) with no network route, except to its model provider if the contract allows it, and no access to the Airlock’s files, memory or credentials, which stay with the routines. The implementation is documented in the GitHub repository.

## Lemma 2 Isolation

Under isolation ( A7 ), for every agent $\alpha ,$ adversarial or not, for every ${ \sf T } \in \mathcal { T } \cup \{ \mathsf { c t r } { \sf 1 } \}$ , and all inputs, worlds and states,

$$
\begin{array} { r } { E _ { \mathsf { T } } ^ { \alpha } ( i , s , s ^ { \prime } ) \implies E _ { \mathsf { T } } ^ { \bullet } ( i , s , s ^ { \prime } ) , \qquad \widehat { E } _ { \mathsf { T } } ^ { \alpha } ( i , w , s , w ^ { \prime } , s ^ { \prime } ) \implies \widehat { E } _ { \mathsf { T } } ^ { \bullet } ( i , w , s , w ^ { \prime } , s ^ { \prime } ) . } \end{array}
$$

Consequently, coverage and fidelity established for the reference executions imply A2 and A5 for every agent.

Proof. Fix an agent �. Consider an execution in which the agent � runs, and in it an invocation of T on input � that starts in the world � and the state �. For each instant � of the invocation, write �(�), �(�) and $r ( \tau )$ for the world, the state and the internal state of the routine at �. By $\left( \mathbf { A } \mathbf { 7 } \right) ,$ nothing the agent does changes these three quantities, and how they can evolve depends only on their current values and on �, not on the agent. The trajectory $\tau \mapsto ( w ( \tau ) , s ( \tau ) , r ( \tau ) )$ is therefore one that a reference execution, started in � and � with the same input, can also produce. Every pair $( \boldsymbol { w } ( \tau ) , s ( \tau ) )$ is thus reached in a reference execution, which proves the implication for $\widehat { E } _ { \top } ^ { \alpha }$ , and projecting on states gives it for $E _ { \mathsf { T } } ^ { \alpha }$ . Coverage and fidelity for the reference executions therefore apply to every pair reached in the actual execution. ■

For a stepwise account of this argument, Appendix G states suficient conditions under which the agent’s computation can be removed while preserving the worlds and states that an invocation can reach. Host and service activity remain possible in the reference executions. The argument uses non-interference, which studies whether the activity of one process can afect another (Goguen and Meseguer, 1982; Rushby, 1992).

The lemma allows us to establish coverage and fidelity by studying the tool together with the services and outside interactions that remain possible when the agent cannot act directly. These conditions then apply to the calls of any agent that uses the system through the Airlock. Combined with the earlier results, they give the guarantees of PC2 for every agent, which Sec. 5.4 states together (Thm. 5).

Under these conditions, changing the agent does not require a new argument about the efects of each tool. The argument still depends on the tool, its input, the host and the external services, whose properties must continue to hold. In Sec. 6.1 we will use this separation when studying how guarantees for individual capability modules can support the tools of diferent contracts.

Isolation is suficient for this conclusion, but safety does not require prohibiting every interaction outside the Airlock. An agent may retain direct access to a service’s API or permission to modify a shared record, provided all efects possible through that access are covered by the models and preserve the required properties, without relying on the agent to choose its actions responsibly. Conversely, removing the agent’s direct access does not resolve unsafe behaviour by a service or another party: protecting a resource against them is the role of custody, which applies complete mediation to the resource rather than to the agent (Proposition 2). The construction is particularly applicable when the system already limits how resources can change, and isolation prevents the agent from bypassing those limits.

## 5.3 Execution and duration of the guarantees

The correspondence arguments above explain how records can remain meaningful as routines and services act on the world. To combine these arguments with internal coherence, we must also establish that the deployed checker, control routine and execution loop behave as required in Sec. 4.1. This brings us back to A1 – A4 . In particular, protecting the proof gate includes protecting the checker and contract, the recorded views, and the association between a proof and the tool and input that actually run. For example, replacing the input after its proof is checked breaks the connection required by A4 , even if the checker itself is sound. The deployed routines must also satisfy the models used to form obligations: A2 covers every state encountered during execution, including outside changes, and A3 requires control to preserve compliance from every compliant active state. Verification must therefore concern the routines that run, under the service and host conditions used to establish these properties. The loop and checker are documented in the GitHub repository (Sec. 7).

Pauses and timing. A pause during a routine is part of that routine’s execution interval. Any changes while it is paused must be covered by its model under A2 and preserve correspondence under A5 . If unrestricted pauses make these properties false, the deployment needs an appropriate bound or protection that remains efective during the pause, such as a lease that expires unless the Airlock renews it (Gray and Cheriton, 1989). For bounded gaps between routines, see Appendix D. A control window bounds the number of control steps rather than elapsed time.

Duration and completion. Thm. 3 guarantees safety up to closing, and throughout a nonclosing execution. It does not imply that a pending request is eventually executed or that settlement succeeds. These are liveness properties, while the guarantees are safety properties (Alpern and Schneider, 1985), the class that a mechanism checking each step can enforce (Schneider, 2000). A deadline is the exception: written as a clause of the contract (Example 1), it is a safety property, and Thm. 5(4) obtains it from anytime compliance, independently of whether the agent makes progress. Likewise, the absence of a further approval before an authorized settlement call (Sec. 4.2) requires the deployment to hold the authority needed to issue it. That authority does not guarantee that the service completes the transfer. Responsibility for unmet conditions and possible remedies are separate matters, discussed in Sec. 8.

Some protections outlast the contract. In the payment ledger of Appendix E, the reservations recorded at closing still bound the charges that arrive afterwards, without the Airlock continuing to run, as long as no new instruction can be issued and the provider keeps respecting its limits.

## 5.4 The guarantees of PC2

We now state the guarantees of PC2 together, in the world and for every agent.

Theorem 5 The guarantees of PC2

Assume A1 , A3 , A4 , A6 and A7 , and A2 and A5 for the reference executions. Then every execution of Alg. 1 satisfies the following, for every agent and every sequence of requests and proofs.

(1) Anytime safety. Safe holds in the world at every instant from the start of the contract until it closes, or at every instant after the start if it never closes.

Moreover, if the contract is in standard form (Example 1), it ends in at most one of the three ways of Thm. 2, and:

(2) Settlement. If it is settled, the goal held in the world when the settlement started.

(3) Termination. If it is terminated, a control step found a termination condition.

(4) Deadline. If it has a deadline � and starts, either settlement or termination is initiated by �.

Proof. We follow the guarantees from the agent to the world.

Every agent. By isolation ( A7 ) and Lemma 2, every efect of a routine, in an execution with any agent, is also an efect of a reference execution. Coverage and fidelity established for the reference executions therefore give A2 and A5 for every agent, and requests and proofs may be arbitrary.

The clauses. The checker accepts only true obligations ( A1 ), the models cover what the routines do ( A2 ), the control step starts the contract compliant and keeps it compliant ( A3 ), and a call runs only for an accepted proof, within its window ( A4 ). These are the hypotheses of Thm. 1, so C holds at every instant from the start.

The world. When the control step starts the contract, the state describes the world, and every later operation preserves this description ( A5 ), so the state describes the world at every instant from the start (Thm. 3). A world described by a compliant state is safe ( A6 , first part). This gives (1).

How the contract ends. In standard form, only the routines of ctrl and settle change the label, and the loop stops once the contract is inactive ( A4 ), so the contract ends in at most one of the three ways (Thm. 2). If it is settled, the accepted proof of the settlement call gives the goal on the state when settlement starts (Thm. 2); the state then describes the world, and a world described by a state with the goal satisfies the goal in the world ( A6 , second part). This gives (2), as in Cor. 4. If it is terminated, the control step terminated it only from a decision reading in End. This gives (3).

The deadline. If the contract has a deadline �, every ongoing state has � ≤ � by the clauses. Since � records current time on the agreed timescale, the contract cannot remain ongoing after �. The label changes only at the end of the settlement or termination routine, so the routine that closed the contract must have started by �. This gives (4). ■

## 6 Composing guarantees

Sec. 4 and 5 have described the guarantees through abstract objects: states, tool models, efects and correspondences with the world. In an application, these objects describe concrete routines, records and services, such as Example 2 or the payment ledger and booking service described in Appendix E. Constructing these descriptions and their proofs anew for every contract would require substantial repeated work, with repeated opportunities to omit an efect or a condition on the service. Much of this work concerns the available operations rather than the particular agreement. The discipline of recording a payment instruction before issuing it, for example, can support contracts with diferent budgets and goals.

We propose to organize this work in reusable capability modules. A module can be developed as a software component whose routines come with state descriptions, tool models, a correspondence with the world and guarantees under stated conditions. A contract can then be built from a combination of these modules, reusing their routines and arguments while specifying the constraints of the particular agreement. Reuse requires preserving the conditions under which the modules were verified, and their combination may require additional arguments. The purpose of this section is to show how such a process can work through a few constructions and examples, rather than to give an exhaustive account of capability composition.

In Sec. 6.1 we define capability modules and explain how to combine them into a state space and a collection of tools on which to build a contract. We give conditions under which their coverage and fidelity guarantees establish A2 and the first line of A5 for the combined tools. The contract still requires its own arguments for the start, the control step and adequacy, alongside the conditions on the checker and execution mechanism. In Sec. 6.5 we bring these arguments together to identify what must be established about the routines, host and services used by a particular contract, and who can provide evidence for it. This produces a tree in which an implication from more specific conditions to a claim is a refinement.

## 6.1 Capability modules

A contract may use several services, each described by diferent state fields and a diferent correspondence to the world. For instance, a contract that makes payments and obtains booking confirmations can combine a payment ledger with a record of confirmations. While a booking request runs, earlier payments may still produce charges. Each module must therefore account both for its own operations and for what can happen while none of them runs. We represent the latter by an idle tool.

## Definition 5 Capability module

A capability module is a tuple $M = ( \mathcal { W } _ { M } , \Sigma _ { M } , \rho _ { M } , \mathcal { T } _ { M } , \mathsf { i d l e } _ { M } )$ , where $\mathcal { W } _ { M }$ and $\Sigma _ { M }$ are sets, $\rho _ { M }$ is a relation on $\mathcal { W } _ { M } \times \Sigma _ { M }$ , and $\mathcal { T } _ { M }$ is a set of tools over $\Sigma _ { M }$ whose routines operate on the system described by $\mathcal { W } _ { M }$ . The tool idl $\mathsf { e } _ { M } \in \mathcal { T } _ { M }$ has no caller-supplied input and waits without intervening. Its efects describe the changes possible during intervals in which no other routine of � runs.

Several modules $M _ { 1 } , \ldots , M _ { n }$ can supply the state space of a contract, for example by taking $\begin{array} { r } { \Sigma = \prod _ { k = 1 } ^ { n } \Sigma _ { M _ { k } } } \end{array}$ and building its tools from their operations, as we describe below.

Here $\mathcal { W } _ { M }$ describes the resource and the services relevant to its operations, while $\Sigma _ { M }$ contains the quantities through which the contract reasons about them. These quantities may be protected records, such as the ledger, or current observations, such as a price the tools would report if asked. An observation of the current price may change while the module waits. A stored quotation remains unchanged, but its correspondence must then describe the quotation and its validity, rather than identify it with the current price. This is the distinction between a possible observation and a recorded reading made in Sec. 3.

idl $\Rightarrow M$ and ctrl. The tool idl $\mathsf { e } _ { M }$ describes what may happen while the module waits without intervening. This can be specified independently of a contract, whereas the contract’s reserved tool ctrl must preserve its particular clause C and respond to its termination conditions. A contract might use a local control routine, say $\mathsf { e x e c } _ { \mathsf { c t r l } _ { M } } ,$ when the necessary intervention depends only on the module’s state. In other cases the decision requires information from several modules: whether to release a payment can depend on a booking and a deadline. Such a decision may require a routine operating on the combined state. We therefore include idle<sub>�</sub> in every module without requiring a local controller.

Local guarantees. Coverage and fidelity can first be established for each module separately. Write $E _ { \mathsf { T } }$ and $\widehat { E } _ { \top }$ for its tools’ state and joint world–state efects, as in Sec. 4.1 and 5.1, with all

(a) Product

![](images/d23a3f420d971ca76d92be2551e5117d9b55c740a253dc254eeb851d628aaa43.jpg)  
Figure 6. Product and extension of capability modules. Circles denote world coordinates, rounded squares denote state coordinates, dotted lines denote correspondence and arrows denote routine actions. (a) The product contains both modules, their lifted tools and the joint idle tool. (b) The outer frame encloses a chosen extension ext $. ( M _ { 1 } \otimes M _ { 2 } )$ of the product in the inner grey frame. It adds the routine ${ \sf T } ^ { \prime } ,$ state � and correspondence $\rho ^ { \prime }$ , with the latter two highlighted in orange. The product’s existing tools are omitted in (b). The extension notation is defined in Definition 7 and used again in Fig. 7.

states and worlds now belonging to that module. For each ${ \sf T } \in \mathcal { T } _ { M }$ , define

$$
\begin{array} { r l } & { \mathsf { C o v } _ { M } ( \mathsf { T } ) : \ \{ \Longleftrightarrow \ \forall i , s , s ^ { \prime } \quad \Xi _ { \mathsf { T } } ( i , s ) \wedge E _ { \mathsf { T } } ( i , s , s ^ { \prime } ) \ \Longrightarrow \ \mathsf { D } _ { \mathsf { T } } ( i , s , s ^ { \prime } ) , } \\ & { \mathsf { F i d } _ { M } ( \mathsf { T } ) : \ \Longleftrightarrow \ \forall i , w , s , w ^ { \prime } , s ^ { \prime } \quad \rho _ { M } ( w , s ) \wedge \Xi _ { \mathsf { T } } ( i , s ) \wedge \widehat { E } _ { \mathsf { T } } ( i , w , s , w ^ { \prime } , s ^ { \prime } ) \ \Longrightarrow \ \rho _ { M } ( w ^ { \prime } , s ^ { \prime } ) , } \end{array}\tag{19}
$$

(20)

with the quantifiers ranging over the tool’s inputs and the module’s spaces. These are the routine conditions of A2 and A5 for one module. For idle, fidelity applies to waiting intervals from states where its model is enabled. Under A7 , these guarantees can be established using the reference efects of Sec. 5.2: Lemma 2 then makes them applicable to every agent.

For example, whoever implements a payment module can prove that its routine records an instruction before sending it, subject to stated conditions on who can change the ledger and use the payment credentials. Another contract can reuse that proof when it uses the same routine under the same conditions. In Sec. 6.3 we give three constructions that make these conditions explicit.

## 6.2 Composing capability modules

To build an environment suited to an application, we need to combine modules that provide its diferent operations and descriptions of the world. For example, a travel application can draw on payment and booking modules, then add an operation relating a purchase to its payment. We consider two constructions, illustrated in Fig. 6. A product puts the modules’ world and state descriptions together and makes their existing tools available on the combined state. An extension adds fields, correspondence conditions or routines involving those components. We first describe how tools on separate modules can be combined, since this is used in both constructions.

Paired tools. Let $U \in \mathcal { T } _ { M _ { 1 } }$ and $V \in \mathcal { T } _ { M _ { 2 } }$ , where the modules have disjoint world and state coordinates. We write $U \otimes V$ for a tool over $\Sigma _ { M _ { 1 } } \times \Sigma _ { M _ { 2 } }$ with input space $I _ { U } \times I _ { V }$ . When both components are non-idle, its routine executes them concurrently in a single invocation and returns when both have returned. If one component is idle, it waits until the other returns.

Pairing two idle tools gives an operation that waits on both components. The model is

$$
\begin{array} { r } { \mathsf { D } _ { U \otimes V } \big ( ( i _ { 1 } , i _ { 2 } ) , ( s _ { 1 } , s _ { 2 } ) , ( s _ { 1 } ^ { \prime } , s _ { 2 } ^ { \prime } ) \big ) \iff \mathsf { D } _ { U } ( i _ { 1 } , s _ { 1 } , s _ { 1 } ^ { \prime } ) \wedge \mathsf { D } _ { V } ( i _ { 2 } , s _ { 2 } , s _ { 2 } ^ { \prime } ) . } \end{array}\tag{21}
$$

Compatibility of efects. Reusing the separate tools’ guarantees requires checking that their combination does not introduce efects outside those for which the guarantees were established. Write $i = ( i _ { 1 } , i _ { 2 } ) , w = ( w _ { 1 } , w _ { 2 } )$ and $s = ( s _ { 1 } , s _ { 2 } )$ , and similarly for primed quantities. The required condition, for all such inputs and coordinates, is

$$
\widehat { E } _ { U \otimes V } ( i , w , s , w ^ { \prime } , s ^ { \prime } ) \Longrightarrow \widehat { E } _ { U } ( i _ { 1 } , w _ { 1 } , s _ { 1 } , w _ { 1 } ^ { \prime } , s _ { 1 } ^ { \prime } ) \wedge \widehat { E } _ { V } ( i _ { 2 } , w _ { 2 } , s _ { 2 } , w _ { 2 } ^ { \prime } , s _ { 2 } ^ { \prime } ) .\tag{22}
$$

The condition concerns the whole paired invocation. If one routine finishes before the other, its efects and guarantees must also cover the remaining wait. When $V = \mathsf { i d } | \mathsf { e } _ { M _ { 2 } } .$ , running � must leave all changes on the second module within its idle efects, and conversely for an idle first component. Disjoint coordinates therefore need not mean that the resources are unafected by each other. For example, a merchant may charge an earlier payment instruction while a booking call runs, provided the payment module’s idle efects account for it. If an interaction introduces changes outside a component’s efects, its description and guarantees must be revised, or the interacting parts must be treated together.

This is the question addressed by rely–guarantee reasoning: the conditions under which one component was verified must be respected by the other components and their environment (Jones, 1983; Abadi and Lamport, 1995). Here (22) states that requirement directly for the efects used in our coverage and fidelity arguments.

Product. A product makes each module’s tools available while accounting for changes on the other modules as they wait. For payment and booking modules, a payment call becomes pay $\otimes \operatorname { i d } \lvert \boldsymbol { \mathrm { e } } _ { \mathrm { b o o k } }$ : its model includes the payment routine’s efects and the changes that the booking can undergo during the payment.

Definition 6 Product modules   
For modules $M _ { 1 } , M _ { 2 }$ with disjoint world and state coordinates, their product $M = M _ { 1 } \otimes M _ { 2 }$   
has $\mathcal { W } _ { M } = \mathcal { W } _ { M _ { 1 } } \times \mathcal { W } _ { M _ { 2 } } , \Sigma _ { M } = \Sigma _ { M _ { 1 } } \times \Sigma _ { M _ { 2 } }$ and   
�<sub>�</sub> ( (�<sub>1</sub>, �<sub>2</sub>), (�<sub>1</sub>, �<sub>2</sub>)) ⇐⇒ �<sub>�</sub> (�<sub>1</sub>, �<sub>1</sub>) ∧ �<sub>�</sub> (�<sub>2</sub>, �<sub>2</sub>).   
Its tools are   
T<sub>�</sub> = {T<sub>1</sub> ⊗ idle<sub>�</sub> : T<sub>1</sub> ∈ T<sub>�</sub> } ∪ {idle<sub>�</sub> ⊗ T<sub>2</sub> : T<sub>2</sub> ∈ T<sub>�</sub> }. (23)   
The idle tool is id ${ \tt e } _ { M _ { 1 } \otimes M _ { 2 } } = \mathsf { i d } | { \tt e } _ { M _ { 1 } } \otimes \mathsf { i d } | { \tt e } _ { M _ { 2 } }$

The product is associative and commutative up to the canonical regrouping and permutation of coordinates: $M _ { 1 } \otimes \left( M _ { 2 } \otimes M _ { 3 } \right) \cong \left( M _ { 1 } \otimes M _ { 2 } \right)$ ⊗ $M _ { 3 }$ and $M _ { 1 } \otimes M _ { 2 } \cong M _ { 2 } \otimes M _ { 1 }$ . These identifications preserve the correspondence and tool models and retain the same component routines. We can therefore write $M _ { 1 } \otimes \cdots \otimes M _ { n }$ without parentheses for any finite nonempty family of modules with pairwise disjoint coordinates.

Extension. Applications may also require records that relate several modules or operations that coordinate their use, for example by chaining calls or executing them concurrently. An extension can add these operations together with new state fields and a correspondence describing what those fields mean. Making the additions explicit identifies what must be established beyond the guarantees already supplied by the modules.

## Definition 7 Extension modules

Let � be a module, � a nonempty set, $\rho ^ { \prime }$ a relation on $\mathcal { W } _ { M } \times \Sigma _ { M } \times Z ,$ , and ${ \mathcal { T } } ^ { \prime }$ a set of tools over $\Sigma _ { M } \times Z$ . The extension $N = \mathrm { e x t } ( M ; Z , \rho ^ { \prime } , \mathcal { T } ^ { \prime } )$ has $\mathcal { W } _ { N } = \mathcal { W } _ { M } , \Sigma _ { N } = \Sigma _ { M } \times Z$ and

$$
\rho _ { N } ( w , ( s , z ) ) \longleftrightarrow \rho _ { M } ( w , s ) \wedge \rho ^ { \prime } ( w , s , z ) .
$$

Its tools are $\{ \widetilde { \mathsf { T } } : \mathsf { T } \in \mathcal { T } _ { M } \} \cup \mathcal { T } ^ { \prime }$ . Each retained tool $\widetilde { \mathsf { T } }$ has the same inputs and routine as T, leaves � unchanged and has model

$$
{ \mathsf D } _ { \widetilde { \mathsf T } } ( i , ( s , z ) , ( s ^ { \prime } , z ^ { \prime } ) ) \longleftrightarrow { \mathsf D } _ { \mathsf T } ( i , s , s ^ { \prime } ) \wedge z ^ { \prime } = z .\tag{24}
$$

The idle tool is $\widetilde { \mathsf { i d } | \mathsf { e } _ { M } }$ , and the fields in � change only through the added routines.

Table 1. Three constructions of capability modules, each illustrated by a diferent example. The world, state, correspondence and tools describe the module, while the supporting conditions establish the guarantees below. A module can combine these constructions.
<table><tr><td>Module element</td><td>Custody (Ex.: payment instructions)</td><td>Formal interface (Ex.: booking responses)</td><td>Domain invariant (Ex.: mailbox retention)</td></tr><tr><td>World  $\mathcal { W } _ { M }$ </td><td>Payment service and history J(ω) of submitted instructions.</td><td>Booking service and history H(w) of issued responses.</td><td>Mail service, including delivery and acknowledgement history.</td></tr><tr><td>State  $\Sigma _ { M }$ </td><td>Register of payment instructions.</td><td>Log of response events.</td><td>Current mailbox contents π(w).</td></tr><tr><td>Correspondence ρM</td><td> $J ( w ) \subseteq s$  Every submitted instruction is registered.</td><td> $\forall e \in s ,$   $e \in H ( w ) \land \mathsf { S p e c } ( e )$  Recorded responses were issued and are permitted.</td><td> $s = \pi ( w ) \land \mathsf { K } ( w )$  The reading is current and pending mail remains available.</td></tr><tr><td>Tools  $\mathcal { T } _ { M }$ </td><td>issue(j) records an instruction, then submits it.</td><td>query(q) requests, authenticates and records a response.</td><td>Read or process messages, with models covering changes to mailbox contents.</td></tr><tr><td>Idle idlem</td><td>Register unchanged. Earlier instructions may still produce charges.</td><td>Log unchanged. The service may continue to evolve.</td><td>Mailbox contents may change, while pending messages remain available.</td></tr><tr><td>Supporting conditions</td><td>Protect register updates H1 Register before submitting and restrict issuance H2.</td><td>Permitted responses (F1) authenticated and recorded faithfully F2.</td><td>Readings agree with the world M1. Routines and outside changes preserve K</td></tr><tr><td>Local result</td><td>Coverage and fidelity (Proposition 2)</td><td>Coverage and fidelity (Proposition 3)</td><td>M2. Fidelity (Proposition 4). Coverage is checked separately.</td></tr></table>

Safe(�). The checker and execution mechanism remain subject to A1 and A4 .

## 6.3 Three capability constructions

We now give concrete conditions under which a module satisfies its local guarantees. The constructions in Tab. 1 distinguish a protected record of instructions, a record of what a service reported, and a description of quantities that may change independently. They can be combined when a resource requires more than one of these arguments.

Custody. Consider a payment account whose credentials are kept by the Airlock. Its payment routine can send the provider an instruction specifying a recipient and a spending limit, with an identifier distinguishing it from other instructions. Before sending the message, the routine adds the instruction to a protected register. This order matters because the provider may charge the account as soon as it receives the message, before any acknowledgement returns. We want the register to account for every instruction that the provider can act on, including those still in progress. This is the part of the payment-ledger argument in Appendix E that concerns custody of the authority to issue payments. The bound on the charges themselves additionally depends on the provider respecting each instruction’s limit.

Let Instr be the set of these payment instructions, $\Sigma _ { M }$ the set of its finite subsets, and $I _ { \mathrm { i s s u e } } = I n s t r$ A state � is the register. Write $J ( \boldsymbol { w } )$ for the history of submitted instructions, including pending instructions from the moment of submission. Define

$$
\begin{array} { r } { \begin{array} { l } { \rho _ { M } ( w , s ) \iff \forall j \in J ( w ) . \ j \in s , } \\ { { \sf D } _ { \mathsf { i s s u e } } ( j , s , s ^ { \prime } ) \iff s ^ { \prime } = s \vee s ^ { \prime } = s \cup \{ j \} , \qquad { \sf D } _ { \mathsf { i d l e } _ { M } } ( s , s ^ { \prime } ) \iff s ^ { \prime } = s . } \end{array} } \end{array}\tag{25}
$$

The module has these two tools. The model of issue includes the register before and after the instruction is added, while idle leaves it unchanged. Entries are retained even when a request times out, since an unacknowledged instruction may still produce a charge.

Proposition 2 Custody

For the module defined by (25), assume that, in every execution of its tools, including idle,

H1 only issue(�) changes the register, by adding � at most once during the invocation. Entries are never removed or modified;

H2 an instruction is submitted only by issue after it is registered. The history � retains all submissions, and no other action adds an instruction or changes its terms.

Then $\mathsf { C o v } _ { M } ( \mathsf { T } )$ and $\mathsf { F i d } _ { M } ( \mathsf { T } )$ hold for $\mathsf { T } \in \{ \mathsf { i s s u e } , \mathsf { i d l e } _ { M } \}$

Proof. H1 limits register changes to those in (25), establishing coverage for both tools. Starting from correspondence, every initial member of � remains registered by H1 . Every new member is registered before submission by H2 , and its entry is retained. Thus $J ( w ^ { \prime } )$ remains contained in $s ^ { \prime }$ throughout either invocation, establishing fidelity. ■

The access restriction in H2 applies complete mediation (Saltzer and Schroeder, 1975) to the authority to issue payment instructions. Keeping the credentials under the Airlock’s control and requiring the provider to accept only authenticated instructions can support this restriction, as in capability-based protection (Dennis and Van Horn, 1966; Miller, 2006). The Airlock’s implementation must protect the register and record each instruction before sending it. The payment provider must preserve its terms and reject submissions made without the required authority. These conditions do not assert that the account balance remains fixed while the module is idle. Charges from earlier instructions may continue, and their bounds require the additional service condition in (E.2).

Formal interfaces. A service can specify which responses it may return to a request and how they relate to that request and to its previous interactions. Such a specification gives a module a basis for modelling calls to the service without describing its implementation. The module’s routine records the responses it receives, and its correspondence connects those records to responses actually issued by the service and permitted by its specification.

For example, a booking request can identify a hotel, dates and a maximum price. The specification may allow either a rejection or a confirmation carrying those dates and a price within the requested limit. Concretely, an implementation could maintain a register of response records, stored in memory or in a database. Each record would contain the request, the returned status and, for a confirmation, the booking identifier, dates and price. The routine authenticates the response and adds its record to the register. The module’s state represents the register’s contents as a collection of such records, so the model can express which additions are permitted without committing to a particular storage implementation. Correspondence then relates each record to a response actually issued by the service. Whether the hotel honours the confirmation is a further property, supported by the reservation condition in Example 3.

The same construction applies to any service with a specified request–response interface. Let Req be the set of requests and Resp the set of response events. An event contains the request, the returned data and any time or historical information used by the specification. Write request(�) for its request and Spec(�) for the statement that the response is permitted for that request. Take $I _ { \mathsf { q u e r y } } = R e q$ and let $\Sigma _ { M }$ consist of finite sets of response events. With $H ( \boldsymbol \omega )$ denoting the history of events actually issued by the service, define

$$
\begin{array} { r l } & { \qquad \rho _ { M } ( w , s ) \iff \forall e \in s . e \in H ( w ) \wedge \mathsf { S p e c } ( e ) , } \\ & { \mathsf { D } _ { \mathsf { q u e r y } } ( q , s , s ^ { \prime } ) \iff s ^ { \prime } = s \vee \exists e . \mathsf { r e q u e s t } ( e ) = q \wedge \mathsf { S p e c } ( e ) \wedge s ^ { \prime } = s \cup \{ e \} , } \\ & { \mathsf { D } _ { \mathsf { i d l e } _ { M } } ( s , s ^ { \prime } ) \iff s ^ { \prime } = s . } \end{array}\tag{26}
$$

The model allows any response permitted by the specification, while correspondence requires each recorded response to have actually been issued. The following proposition separates the conditions needed on the service from those needed on the routine that records its responses.

## Proposition 3 Formal interface

For the module defined by (26), assume that � retains its events and that, in every execution of its tools, including idle,

F1 every response event issued by the service satisfies Spec;

F2 only query(�) changes the log. It either leaves it unchanged or appends one event � issued for �, after authenticating its origin and recording its contents correctly. Entries are never removed or modified.

Then $\mathsf { C o v } _ { M } ( \mathsf { T } )$ and $\mathsf { F i d } _ { M } ( \mathsf { T } )$ hold for $\mathsf { T } \in \{ \mathsf { q u e r y } , \mathsf { i d } | \mathsf { e } _ { M } \}$

Proof. By F2 , any change appends one response event for the current request, and F1 makes that event satisfy Spec. This proves coverage. Initially valid entries remain in �, since history is retained. Each new entry is an issued event by F2 and satisfies Spec by F1 , proving fidelity. During idle the log is unchanged and its entries remain valid. ■

If no response has arrived, or the call times out without recording one, the log remains unchanged, which the model permits. An error returned by the service can instead be a response allowed by Spec. Condition F1 requires the actual responses to obey that specification, for example by respecting the requested dates and price limit. This is a property of the service, for which verification or testing can provide evidence (Tretmans, 2008; Bornholt et al., 2021; Disselkoen et al., 2024). Condition F2 concerns the Airlock’s routine: it must authenticate the response and record it against the correct request. Authentication establishes who sent the response, but cannot by itself establish that its contents satisfy the specification or that the hotel will honour the booking (Zhang et al., 2016, 2020).

Domain invariants. Interpreting a service’s records often relies on properties of the system in which the service operates. A booking confirmation can stand for a reservation still held only if the hotel maintains that reservation under the agreed conditions. Such properties can support many diferent contracts using the same service. We express them by a predicate K on worlds, and ask that the module’s routines and the outside interactions that remain possible preserve K whenever it holds initially. This is the sense in which K is a domain invariant. It describes a relationship that the system must maintain, which may involve facts absent from the module’s state.

Consider a module whose state is a measurement $\pi : \mathcal { W } _ { M } \to \Sigma _ { M }$ , with $\Sigma _ { M } = \pi ( \mathcal { W } _ { M } )$ . Here $s ~ = ~ \pi ( \boldsymbol { w } )$ is the current, possibly unobserved state, as in Sec. 3. For example, a module that monitors a mailbox can take $\pi ( \boldsymbol { w } )$ to be its current contents. Write Pending $( w , m )$ when message � has been delivered to the mailbox but has not yet been acknowledged by the module’s processing routine, and Available $( \boldsymbol { w } , m )$ when it can still be retrieved from the mailbox. A retention condition on the mail service is

$$
\mathsf { K } ( w ) \quad \Longleftrightarrow \quad \forall m . \mathsf { P e n d i n g } ( w , m ) \implies \mathsf { A v a i l a b l e } ( w , m ) .
$$

This condition allows new messages to arrive and acknowledged messages to be removed, while requiring pending messages to remain available. Delivery and acknowledgement belong to

the world’s history, so a correct reading of an empty mailbox alone does not establish that no messages are pending. That conclusion also uses K. The module’s guarantees therefore depend on the routines and mail service preserving this retention condition.

We include both measurement agreement and the domain invariant in the correspondence:

$$
\rho _ { M } ( w , s ) \longleftrightarrow s = \pi ( w ) \wedge \mathsf { K } ( w ) .\tag{27}
$$

The tools retain their models D and their execution efects $E _ { \mathsf { T } }$ and $\widehat { E } _ { \top }$ . Coverage requires $\Xi _ { \mathsf { T } } ( i , s ) \wedge E _ { \mathsf { T } } ( i , s , s ^ { \prime } )\implies \mathsf { D } _ { \mathsf { T } } ( i , s , s ^ { \prime } )$ , as in (19), and must be established for the chosen models. The role of K is to justify the additional relationships asserted by correspondence. To establish fidelity, we check that the actual joint efects $\widehat { E } _ { \top }$ preserve K while the state continues to agree with �. The following proposition expresses these conditions and shows how K can then support an adequacy argument.

## Proposition 4 Domain invariant

For a module with correspondence given by (27), assume that

M1 in every execution of its tools, including idle, the state at every instant is � of the world at that instant;

M2 for every $\mathsf { T } \in \mathcal { T } _ { M }$ and all inputs, worlds and states,

$$
\Xi _ { \mathsf { T } } ( i , s ) \wedge \mathsf { K } ( w ) \wedge \widehat { E } _ { \mathsf { T } } ( i , w , s , w ^ { \prime } , s ^ { \prime } ) \implies \mathsf { K } ( w ^ { \prime } ) .
$$

Then $\mathsf { F i d } _ { M } ( \mathsf { T } )$ holds for every $\mathsf { T } \in \mathcal { T } _ { M }$ . For predicates C on $\Sigma _ { M }$ and Safe on $\mathcal { W } _ { M }$ , adequacy follows if

$$
\forall w \in { \mathcal { W } } _ { M } \qquad \mathsf { K } ( w ) \wedge \mathsf { C } ( \pi ( w ) ) \implies { \mathsf { S a f e } } ( w ) .\tag{28}
$$

Proof. Consider an enabled invocation starting from $\rho _ { M } ( w , s )$ and any joint efect $\widehat { E } _ { \mathsf { T } } ( i , w , s , w ^ { \prime } , s ^ { \prime } )$ Initial correspondence gives $\mathsf { K } ( \boldsymbol \omega )$ , so M2 yields $\mathsf { K } ( \boldsymbol { w } ^ { \prime } ) _ { : }$ , while M1 gives $s ^ { \prime } = \pi ( w ^ { \prime } )$ . Thus $\rho _ { M } ( w ^ { \prime } , s ^ { \prime } )$ holds, proving fidelity. Finally, correspondence and C(�) imply K(�) and C(�(�)), so (28) gives Safe(�). ■

Conditions such as M2 express what the system relies on about its environment, as domain assumptions do in requirements engineering (Jackson, 1995; Zave and Jackson, 1997). In the mailbox example, M2 requires pending messages to remain available throughout processing and waiting. A routine that removes a message must therefore have acknowledged it by the time removal takes efect. Preservation of K can be proved for a specified service whose operations enforce these conditions. Applying that proof to an actual service also requires evidence that its implementation and the permitted outside interactions satisfy the specification (Lee, 2016).

Conditions on outside changes also inform the choice of $\mathsf { D } _ { \mathsf { T } }$ . For example, a bound on how rapidly a price can change must account for the whole duration of an invocation or idle interval. An unbounded idle period can require a much broader model, and the contract may then be unable to authorize an operation. Such coverage arguments concern the possible efects on the state, alongside the invariant-preservation argument above.

Another example concerns the expert tools introduced in Sec. 3.5.2, such as an LLM asked to assess whether a hotel is suitable. An expert module can describe the history of these assessments, with � extracting the judgment returned by a particular invocation. A state field then denotes that identified assessment, so observing it again retrieves the same result even when the procedure could give a diferent answer on a new call. The correspondence establishes which procedure produced the judgment. A contract using the module must additionally specify what reliance on that judgment means for its goal or safety property, which is a question of adequacy.

Table 2. Forms of interaction between a capability module and the world. Each row relates the changes a model must cover to possible contract responses and approaches from the literature.
<table><tr><td>Interaction with the world</td><td>What models cover</td><td>Contract response</td><td>Related approaches</td></tr><tr><td>Protected from outside changes E.g. an instruction register</td><td>No outside changes to the protected fields.</td><td>Rely on these records when authorizing later operations.</td><td>Complete mediation and custody (Saltzer and Schroeder, 1975)</td></tr><tr><td>History only grows E.g. a log of service responses</td><td>Additions to the history, with past events retained.</td><td>Use facts about the past that survive further additions.</td><td>History variables (Abadi and Lamport, 1991)</td></tr><tr><td>Changes within an agreed bound E.g. withdrawals by other account holders</td><td>Every debit permitted by the account&#x27;s shared withdrawal limit.</td><td>Reserve enough margin for those debits.</td><td>Rely-guarantee reasoning (Jones, 1983)</td></tr><tr><td>Changes at a bounded rate E.g. charges for a</td><td>Change bounded by a rate and the duration of the operation.</td><td>Keep a margin and intervene before it is exhausted.</td><td>Controlled invariants, barrier functions and Simplex (Blanchini, 1999; Ames et al.,</td></tr><tr><td>metered service Holds until expiry E.g. a temporary hotel</td><td>The commitment remains valid until its expiry.</td><td>Complete the operation in time, renew, or release.</td><td>2019; Sha, 2001) Leases (Gray and Cheriton, 1989)</td></tr><tr><td>reservation May change beyond our control</td><td>Each possible outcome and its consequences.</td><td>Allow a compliant way to continue or terminate</td><td>Demonic nondeterminism (Dijkstra, 1975)</td></tr><tr><td>E.g. a flight cancellation Responds to a request E.g. a booking confirmation or rejection</td><td>Every response allowed by the specification.</td><td>after each outcome. Prove admissibility for every permitted</td><td>Formal interfaces and conformance testing (Tretmans, 2008)</td></tr></table>

The payment and booking examples of Appendix E combine these constructions. In the ledger, custody supports $u _ { j } \leq r _ { j } .$ , while the provider’s charge bound supports $a _ { j } \leq u _ { j }$ . For bookings, recording an authentic confirmation supports the historical claim, while the hotel’s commitment supports the claim that the room remains held. These conditions belong to diferent parties even when they support one correspondence. Tab. 2 places these constructions in the broader setting of interactions with the world, from protected records to changing quantities and service responses. It connects what the tools must model, including while idle, to the choices available to a contract and to approaches developed in the literature.

## 6.4 From local to global guarantees

The preceding constructions identify conditions on routines, records and services that support a module’s guarantees. Once coverage and fidelity have been established for each module, we can combine them for a contract whose state is their product. For a tool that operates on one component, the proof must account for that operation and for the other components while they wait. More generally, the same argument applies to a combined tool when each component’s model and efects cover the whole invocation. A control routine that cannot be decomposed in this way needs a separate argument over the full state.

## Theorem 6 From local to global guarantees

Let $M _ { 1 } , \dots , M _ { n }$ be capability modules, with $n \geq 1 .$ , and let K have state space $\begin{array} { r } { \Sigma = \prod _ { k = 1 } ^ { n } \Sigma _ { M _ { k } } } \end{array}$ Set $\begin{array} { r } { \mathcal { W } = \prod _ { k = 1 } ^ { n } \mathcal { W } _ { M _ { k } } } \end{array}$ and $\rho ( w , s ) \quad \Longleftrightarrow \quad { \textstyle \bigwedge } _ { k = 1 } ^ { n } \rho _ { M _ { k } } ( w _ { k } , s _ { k } )$ Suppose each contract tool, including control, is a tensor of tools from the $M _ { k }$ , with the conjunction model of (21).

Assume the �-component form of (22) for every such tool. Then

$$
\bigwedge _ { k = 1 } ^ { n } \bigwedge _ { \mathsf { T } \in \mathcal { T } _ { M _ { k } } } \mathsf { C o v } _ { M _ { k } } ( \mathsf { T } ) \implies \ @ \mathsf { \Pi } _ { k = 1 } ^ { \prime } , \qquad \bigwedge _ { k = 1 } ^ { n } \bigwedge _ { \mathsf { T } \in \mathcal { T } _ { M _ { k } } } \mathsf { F i d } _ { M _ { k } } ( \mathsf { T } ) \implies ( 1 5 ) .\tag{29}
$$

Proof. An enabled tensor has an enabled component on each module, because a witness to its conjunction model supplies a witness to every component model. By (22), each joint efect projects to an efect of that component. Local coverage therefore gives each conjunct of the tensor model, proving coverage. Local fidelity preserves each conjunct of $\rho ,$ which proves (15), the first line of A5 . ■

The result explains what can be reused when a contract combines, for example, payment and booking. The proofs about recording payment instructions and booking responses can be established separately. To use them together, we check that running either routine does not introduce changes excluded by the other module’s efects. Once this compatibility holds, the proofs establish coverage and fidelity for the combined tools, without a new proof for every payment–booking combination. The second line of A5 , the start, is a property of the contract’s control step (Sec. 5.1).

The theorem is stated using execution efects � and ${ \widehat { E } } .$ To obtain guarantees independent of the requesting agent, its argument can first be applied to the reference executions of Sec. 5.2, with the same compatibility condition on their efects. Under A7 , Lemma 2 transfers the resulting coverage and fidelity to executions with any agent. This identifies the separate roles of composition and isolation: composition combines the component arguments, while isolation makes the resulting argument apply regardless of the agent’s computation and requests.

A contract can also restrict an existing tool by a precondition, as in Sec. 3.5.2. The new model is the conjunction of that precondition with the original model. Whenever it is enabled, the precondition holds and the original tool is enabled, so the original coverage proof also establishes coverage of the restricted model. The routine and correspondence have not changed, so the same fidelity proof applies. The restriction thus changes which calls can be authorized while retaining the established guarantees for those calls.

Checking an extension. We want to reuse the guarantees of an existing module when adding state fields, correspondence claims or routines. To do so, we first establish that the existing routines retain their guarantees in the extended system, then check that they preserve the added correspondence and that the new routines satisfy the requirements on the full state. The completed-purchase register introduced above illustrates why these checks are needed. An existing booking routine leaves this register unchanged, but that alone does not establish the added correspondence $\rho ^ { \prime }$ . If an entry means that a confirmation was issued, later cancellation need not invalidate it. If it means that the booking is still held, cancellation can make it false even though the entry has not changed.

More precisely, the efects of each existing routine, projected onto $\mathcal { W } _ { M }$ and $\Sigma _ { M } .$ , must remain within its original efects, and � must remain unchanged. Their coverage proofs then establish the lifted models in (24), and their fidelity proofs preserve $\rho _ { M }$ . For each such tool, including idle, we additionally check that an enabled invocation starting from $\rho _ { M } \wedge \rho ^ { \prime }$ preserves $\rho ^ { \prime }$ . For an added routine, such as the purchase routine, coverage must hold on the full state $\Sigma _ { M } \times Z$ and fidelity must preserve both correspondences. An added control routine must also satisfy A3 for the chosen contract clause. These checks allow an extension to reuse the original proofs while establishing the additional claims it makes.

The conditions used in these proofs provide a way to organize the argument for the complete contract. Write $L _ { k }$ for the supporting conditions of module $M _ { k }$ , such as the register and issuance requirements H1 – H2 for payments or the response and recording requirements F1 – F2 for bookings. Suppose they establish its local guarantees for reference executions, and check compatibility in (22) for those same efects. Under the remaining construction hypotheses of Thm. 6, composition followed by isolation gives

![](images/2d123c6071105caa094873759a5547b67d3e5a9644be543fe64061f2040161d4.jpg)  
Figure 7. An illustrative assurance case for a recursively constructed contract environment. The arrows connect separate module guarantees: ${ \cal M } = { \cal M } _ { 1 } \otimes { \cal M } _ { 2 } \otimes { \cal M } _ { 5 }$ , with $M _ { 5 } = \mathrm { e x t } ( M _ { 3 } \otimes M _ { 4 } )$ . The direct branch from $M _ { 3 } , M _ { 4 }$ to $M _ { 5 }$ combines the product conditions of Thm. 6 and the extension checks described in the text. Each coverage or fidelity claim concerns all tools of that module, including idle, for reference executions. Together with isolation, the guarantees of � yield A2 and A5 through Lemma 2. The colour tags record a proof under supporting conditions, responsibility for meeting a condition, measures to reduce the risk of failure, or acceptance of that risk. The assignments are illustrative and may difer between coverage and fidelity of the same module.

$$
L _ { 1 } \land \cdot \cdot \cdot \land L _ { n } \land \ @ \mathbb { Z } \land { \mathrm { c o m p a t i b i l i t y ~ o f ~ r e f e r e n c e ~ e f f e c t s ~ \Longrightarrow ~ \ @ ~ \{ \cdot \} ~ ( 1 5 ) . } }\tag{30}
$$

Fig. 7 illustrates how this argument can be repeated within a module. The contract environment is the product $M _ { 1 } \otimes M _ { 2 } \otimes M _ { 5 } ,$ , where $M _ { 5 } = \mathrm { e x t } ( M _ { 3 } \otimes M _ { 4 } )$ . In the figure, $\mathsf { C o v } _ { N }$ and $\mathsf { F i d } _ { N }$ abbreviate the coverage and fidelity requirements for every tool of a module $N _ { z }$ , including idle, for reference executions. We first combine the guarantees of $M _ { 3 }$ and $M _ { 4 }$ , then check the additions that produce $M _ { 5 }$ . Once these checks are complete, $M _ { 5 }$ supplies guarantees in the same form as the other modules and can be used in the outer product. Each product step requires the hypotheses of Thm. 6, and A7 transfers the resulting guarantees to every agent.

The colours illustrate one possible assessment of the conditions supporting this construction. They apply separately to coverage and fidelity, while the module boxes remain white. The internal claims are proved from their dependencies, while the leaf claims identify where responsibility is assigned, risk is mitigated or risk is accepted. Coverage and fidelity of the same module may therefore receive diferent treatments. These tags describe how a condition is handled, and a diferent use of the same module may require a diferent assignment. In Sec. 6.5 we explain how to organize such an assessment as an assurance case, connecting the mathematical argument to evidence and responsibilities.

![](images/03e44e6f7711a28d4c8c1eb90c8dd3187537ee84c5fced3030d9ec5203789d8d.jpg)  
Figure 8. An illustrative assurance case connecting assumptions, supporting conditions and responsibilities. The first level gives the joint premises of Thm. 3. Isolation connects reference proofs to arbitrary agents. The four groups refine fidelity through the lettered conditions of Propositions 2 to 4, some of which also establish coverage. Teal denotes a proof under its supporting conditions. Grey under adequacy records acceptance of the risk that the chosen scope of protection is incomplete. The colour tags have the same meanings as in Fig. 7, while their assignments and the named roles are illustrative and may be combined or changed for a particular system. Dashed branches indicate patterns to instantiate for its correspondence.

Reusing module arguments reduces repeated verification, although compatibility and extension checks still depend on the combination. It also helps assess a change: if a payment provider changes its access policy, we can follow the branches from that condition to the contracts whose proofs use it. The tree identifies the afected arguments. Whether the resulting system failure can also afect other modules depends on their interactions, particularly on whether compatibility still holds.

Adequacy. The theorem establishes coverage and fidelity. The implication $\rho ( w , s ) \land \mathsf { C } ( s ) \implies$ Safe(�) in A6 still concerns the contract as a whole. Its correspondences may be local while its protection involves several resources. With two payment providers under one budget, for instance, $a ^ { 1 } ( w ) \leq b ^ { 1 } ( s )$ and $a ^ { 2 } ( w ) \leq b ^ { 2 } ( s )$ , together with $b ^ { 1 } ( s ) + b ^ { 2 } ( s ) \leq B ,$ , entail $a ^ { 1 } ( w ) + a ^ { 2 } ( w ) \leq$ �. The parties must also validate that this formal protection is the one they intend. The same distinction applies to the goal and (18).

Beyond products. This construction uses disjoint coordinates for the resources and their states. More general forms of composition can support shared resources or guarantees between interacting components. Separation logic and resource algebras provide ways to describe shared resources with separated permissions (O’Hearn et al., 2001; Reynolds, 2002; Jung et al., 2015). Interface theories and assume–guarantee contracts organize interactions through specified interfaces (de Alfaro and Henzinger, 2001; Benveniste et al., 2018), while layered verification establishes a component against the specification of the layer below (Gu et al., 2015). These approaches could support further capability constructions. Here the product and extension identify the proof obligations for the cases developed above.

## 6.5 From assumptions to responsibilities

The preceding constructions allow a contract to reuse its modules’ guarantees. To apply them, we need to identify how their supporting conditions are met by the routines and services involved. For a payment module, this includes how instructions are recorded, who holds the credentials and which submissions the provider accepts. Following these dependencies identifies what to check and who can supply the evidence. Organizing the argument in this way produces an assurance case: a structured argument explaining why a system should satisfy a specified property, supported by evidence for the conditions on which it relies (Kelly and Weaver, 2004; Bloomfield and Rushby, 2020).

Fig. 8 organizes this assessment for the complete contract. Its first level gives the six assumptions of Thm. 3, with their requirements and the parties expected to support them. The module arguments of Fig. 7 establish coverage and fidelity, while the other branches concern the checker, initialization and control, execution, and adequacy. Isolation connects these arguments to arbitrary agents: by Lemma 2, the reference-execution guarantees apply regardless of the agent’s computation and requests. The Airlock’s provider establishes this restriction on access, as described in Sec. 5.2.

The colours distinguish how the parties propose to support a condition or address the possibility that it fails:

• Proof. A mathematical argument establishes a property from specified premises. For example, the routine can be proved to register each instruction before submitting it. Its assumptions about storage and execution remain explicit dependencies.

• Assigned responsibility. A named party commits to maintaining a condition, such as restricting use of the payment credentials or honouring a reservation. The tree records who must support that commitment with evidence and who is responsible for failures.

• Mitigation. Measures such as testing, independent checks or monitoring reduce the risk of a failure or its consequences. For a service response, this could include checking its fields against the request before recording it. The remaining uncertainty must still be assessed.

• Accepted risk. The parties identify a possibility of failure or a gap in the protection and agree to proceed despite it. For a theorem’s premise, this records the limit of their confidence that it holds. For the scope of Safe, it records the risk that the specified protection does not cover everything they intended.

The four tags retain these meanings throughout the assessment, while their placement in Figs. 7 and 8 illustrates possible choices. They may coexist, for example when a service provider accepts responsibility for a condition while monitoring supplies further evidence about whether it holds. In Fig. 8, fidelity is green because it is derived from the conditions below it. The grey box under adequacy records a diferent decision: the parties accept that the scope of protection expressed by Safe may be incomplete, even when the adequacy implication itself has been proved. The white annotation about the meaning of fields identifies a specification choice.

The expanded part follows fidelity, whose supporting conditions often involve several parties. It groups them by the questions to address when interpreting and maintaining the correspondence. The labels H1 – H2 , F1 – F2 and M1 – M2 connect these branches to the conditions in the preceding propositions. A module may combine several of these arguments.

Custody. We ask who can change the protected records and authorize the operations they describe. In the payment module, the routine registers each instruction before submission, and the provider accepts submissions only through the protected authority. Conditions H1 – H2 express these requirements. Proposition 2 shows that they establish coverage and fidelity. Their justification combines an argument about the routine with evidence about access permissions and the provider’s acceptance of instructions.

Formal interfaces. When a module records a service’s response, we need to know what the service may report and how the routine receives and records it. Conditions F1 – F2 separate these questions. A booking service must return permitted responses, and the receiving routine must authenticate them and associate them with the correct requests. Proposition 3 then establishes coverage and fidelity. The service provider supports the first condition, while the receiving routine’s implementer and operator support the second.

External behaviour. Some claims require a service to maintain a condition after replying, or while the module waits. A hotel must retain an unexpired reservation, and a mail service may be required to retain unacknowledged messages. The predicate K of Proposition 4 expresses such domain invariants. Its preservation, together with agreement between the state and its measurement, establishes fidelity. Coverage also requires an account of possible state changes. These arguments may rely on service commitments, verified rules or bounds on outside interactions, as illustrated in Tab. 2.

Agreed meaning. Before checking whether a field describes the world correctly, we must specify what it describes. A booking field may record that a confirmation was issued or assert that a room is still held. These choices require diferent correspondences. In the measurement construction, choosing � specifies the reading, while M1 requires the state to agree with it. For an expert tool, the record can identify the procedure and invocation that produced a judgment. The parties decide what reliance on that judgment means for the protection or goal, connecting correspondence to adequacy and to (18).

For a payment module, the resulting assurance case can connect its coverage and fidelity proofs to a checked implementation, an assessment of access permissions and evidence about the provider’s acceptance of instructions. The tree records how these pieces support the overall guarantee and makes any unsupported conditions visible. The propositions supply the mathematical implications, while the evidence supports applying their premises to the routines and services that will actually be used (Hawkins et al., 2011). This includes specifications and interactions with external components, which are known sources of failures even in verified systems (Fonseca et al., 2017).

The work can be reused along with the module: a new contract checks that its supporting conditions still apply and that the composition and any extensions meet the requirements of Sec. 6.4. It then uses the existing proofs and evidence while supplying its own clause, control and adequacy arguments. If a payment provider changes its access policy, the dependencies identify which custody arguments and contracts need review. In Sec. 8.3 we discuss recording these responsibilities and agreeing on responses to failures. The same assessment supports completion, where the correspondence must justify the world goal when settlement begins.

## 7 Mechanization

We provide a Lean 4 mechanization of PC2 (de Moura and Ullrich, 2021), available in the GitHub repository. The repository contains the formal definitions and checked proofs, identifies the corresponding results in this paper and makes their assumptions explicit. It also documents the implementation and provides instructions for reproducing the checks.

## 8 Practical use and remaining risks

The preceding sections establish how PC2 can enforce a contract and how its guarantees depend on the routines, services and host that support execution. Using such a contract also requires the parties to understand what they are agreeing to, the agent to find proofs at an acceptable cost, and arrangements for the risks that remain assumed. These questions connect the construction to work on specification, automated reasoning and cooperation between parties with diferent interests. We discuss possible ways to address them, and what would need to be evaluated. Developing and validating these arrangements is outside the scope of the present work.

## 8.1 Writing and understanding agreements

An enforceable agreement need not be the agreement its parties intended. A travel contract might require a flight and a hotel without requiring their dates and destinations to match. Every accepted call could then satisfy the clauses while the resulting trip is unusable. The mathematical adequacy condition in A6 connects C to a specified Safe, but deciding whether Safe captures the intended protection remains a separate task. This is closely related to the dificulties of specifying objectives for optimizing systems and of writing complete agreements (Manheim and Garrabrant, 2018; Hadfield-Menell and Hadfield, 2019).

A possible response is to assist the parties with formalizer agents, building on work on autoformalization (Wu et al., 2022). A formalizer would draft clauses from the parties’ intentions, explain the resulting clauses in ordinary language, and search for compliant states that expose an unintended interpretation. For the travel example, it could present a trip whose bookings satisfy the individual requirements but cannot be used together. These examples give the parties concrete questions to resolve before signing. Capability modules and previously reviewed contract templates can also reduce the amount of specification that must be written afresh. Once $\rho ,$ C and Safe have been specified, formal reasoning can establish their mathematical relationship. Reviewing the explanations and counterexamples serves the diferent purpose of checking that these are the relationships the parties want.

Expert judgments. Some requirements, such as whether a hotel suits a client’s description, depend on the experts introduced in Sec. 3.5.2. A fixed expert procedure gives these requirements an agreed interpretation, but its judgments may be influenced by how evidence is presented. A hotel description could be written to obtain a favourable assessment without corresponding to a desirable hotel. The parties can test proposed experts on borderline and adversarial examples, specify which evidence they receive, and limit repeated attempts to obtain a favourable verdict. When an assessment is intended to remain binding, the contract can retain it and specify when new evidence permits reconsideration. These choices make the decision procedure more predictable and harder to manipulate. Their efectiveness, and whether the procedure captures the intended judgment, require evaluation beyond checking the proofs that use its answers.

Negotiation. The same review matters when an agreement is negotiated. A party may propose clauses that are formally consistent but unexpectedly favourable to itself, so the final agreement must be assessed as a whole. Choreographic and negotiation languages (Gopinathan et al., 2026; Wyse et al., 2026) provide ways to organize how agents reach agreements. Connecting them to PC2 would require their outputs to determine the clauses, tools and other components of the contract that will actually be enforced. Formalizers could support this process by explaining how a proposed revision changes the permitted outcomes.

## 8.2 Finding proofs and using contracts in practice

A request without an accepted proof cannot execute, so the ability to find proofs determines whether the agent can carry out its task. Control can still enforce stopping and refund conditions under the timing assumptions of Sec. 5.3. There is no general procedure that decides all obligations expressible in the underlying logic, and proofs that exist may still be dificult to find before their control window expires. The benefit of the construction is that it requires proofs about proposed calls and the tool models, without requiring a proof of the agent’s reasoning or its entire future behaviour. An individual obligation can nevertheless involve substantial reasoning about hidden state and possible outcomes.

Capability modules ofer a way to reduce this work. A module can provide lemmas about the efects of its tools, which can be reused across calls and contracts. Common contract templates could then reduce many requests to familiar arithmetic or finite checks, while retaining the possibility of more general clauses. For example, a lemma about a payment routine’s maximum charge can be reused when proving that a particular request preserves a budget. A prover can combine such lemmas with Lean’s tactics and learned proof search (Ren et al., 2025; Lin et al., 2025; Wang et al., 2025). It can run as an assistant to the acting agent, since the correctness guarantee depends on checking the resulting proof rather than on which system produced it. The choice of control window also matters: a proof that tolerates several control steps may remain usable despite the time required to construct and check it.

Confidentiality during inference. Access restrictions must also account for how the agent is run. If it sends its observations to an external inference service, that service receives the information in those requests even when every tool call is proof-gated. The parties therefore need to decide whether the provider is an intended recipient. Where it is not, possible arrangements include inference on a machine under the required access controls or within a confidential computing environment (Chrapek et al., 2025). The permitted recipients and the protections ofered by the chosen arrangement must agree with the confidentiality requirements of the contract.

Evaluation. An evaluation of practical use should measure whether contracts capture the intended task, whether the required proofs can be found in time, and whether the resulting system completes useful work at an acceptable total cost. The first question calls for adversarial examples that separate intended outcomes from formally compliant ones. The second calls for measurements of proof-search and checking time, reuse of module lemmas, and requests that expire before authorization. The third includes the cost of constructing the contract, operating control and using external services, compared with other safeguards on the same tasks. Successful task completion must be reported alongside violations and rejected calls, so that rejecting everything cannot count as a successful defense. Such an evaluation would test the proposed use of formalizers and provers, rather than infer their efectiveness from the safety theorem.

## 8.3 Responsibility and consequences

The assurance analysis of Sec. 6.5 identifies what supports a guarantee and what remains assumed. The parties must also decide how to proceed with those assumptions. Before signing, they can record each premise that remains, who accepts responsibility for it, the evidence available, and the response agreed for a failure. For the payment ledger of Appendix E, this includes the reservation discipline of the implementation and the charge limits enforced by the payment service. An agreement to answer for a service failure gives the parties a basis for allocating its consequences. It does not replace the evidence that the service respects its limits.

Where failures can be established from records, a contract could include a remedy with a decidable trigger and an available execution routine. Reconciliation of payment instructions against service statements might, for instance, trigger a refund from reserved funds. Deposits or insurance could provide other ways to allocate losses, depending on what events they cover, what evidence establishes a claim, and whether the promised compensation is available. Designing these arrangements is a further question beyond the enforcement mechanism studied here. An automatic remedy itself needs the same coverage, fidelity and adequacy analysis as any other tool. Compensation addresses a consequence of failure, while the safety theorem concerns whether the failure can occur at all. If the available observations cannot distinguish a failure, its resolution requires additional evidence or a risk the parties explicitly accept.

Consequences for others. An agreement may serve its parties while harming others. Credible commitments can sustain harmful coordination or make threats more credible (Hammond et al., 2025). Access to an Airlock could therefore require additional approval or disclosure for certain contracts, alongside restrictions on the resources and operations it provides. These measures raise questions about authority, confidentiality and protection of people outside the agreement. PC2 can enforce agreed requirements, but deciding which agreements should be permitted requires a broader institutional account.

Participation and economic agency. Enforceable commitments could allow people and smaller organizations to delegate to providers whose systems they cannot inspect, while retaining specified limits on the authority they grant. Whether that possibility translates into broad participation also depends on the cost of writing contracts and finding proofs, access to suitable capability modules, and the terms on which Airlock providers ofer their services. These questions connect to economic agency and the distribution of AI’s gains (Jacobs and Imas, 2026), and to arguments that access must be accompanied by the infrastructure needed to benefit from AI (Gabriel and Kasirzadeh, 2026). Reusable modules and reviewed templates could lower some of these costs. Fair terms and protection for people outside an agreement require additional institutional arrangements.

## 9 Related work

PC2 brings together three lines of work: restrictions on what an agent can execute, proofs supplied by an untrusted requester, and arguments connecting a formal specification to the system in which it is used. Tab. 3 summarizes the enforcement mechanisms. We first explain how proof-carrying changes the task of establishing safety, then compare recent safeguards for agents and the verification, specification and cooperation literature on which the construction builds.

From safety constraints to call obligations. To use an authorization policy for state safety, its author must establish that every permitted call preserves the intended constraint. As tools interact with hidden information, external services and changing state, this can require substantial reasoning about all the situations in which the policy allows execution. A policy can be too restrictive, excluding safe calls, or leave safety gaps despite being enforced correctly.

Table 3. Comparison by feature, with representative examples under each approach. Formal guarantees depend on each construction’s assumptions about observations, models and execution. A ✓ means that the feature is provided, and a ✗ that it is not provided by the mechanism itself. For PC2, the settlement row concerns a contract with a goal-gated settlement tool.
<table><tr><td>Question Examples</td><td>Prompts and learned monitors Prompt delimiters and attack detectors</td><td>Policy enforcement CaMeL (Debenedetti</td><td>Model-based shields Shield synthesis</td><td>PC2 This work</td></tr><tr><td>What allows a tool call</td><td>evaluated in AgentDojo (Debenedetti et al., 2024) The model follows</td><td>et al., 2025) and FORGE (Palumbo et al., 2026) An author-supplied</td><td>(Bloem et al., 2015) and shielded RL (Alshiekh et al., 2018) An analysis of</td><td>A supplied proof</td></tr><tr><td>to run?</td><td>instructions, or a learned monitor accepts the input or call</td><td>rule permits the call, given its arguments, history or data sources</td><td>possible system behaviour identifies the action as safe Safety formulas</td><td>establishes the obligation derived from the contract and tool models State constraints</td></tr><tr><td>How are requirements written?</td><td>Instructions in ordinary language or criteria learned from examples</td><td>Executable rules on calls, past actions and information flow</td><td>supported by the chosen system model and analysis</td><td>from which per-call obligations are derived, even when those obligations are undecidable</td></tr><tr><td>What is established about the call&#x27;s effects?</td><td>No formal guarantee covering all possible outcomes of an accepted call</td><td>Calls and recorded events satisfy the policy. Safety of their consequences needs a further argument</td><td>Every behaviour covered by the safety analysis satisfies the specified safety property</td><td>Every modelled effect preserves the contract, despite hidden state and permitted changes before execution</td></tr><tr><td>Are restrictions guaranteed even for a malicious agent?</td><td>X Resistance to attacks is empirical</td><td>√The specified policy is enforced</td><td>√Safety holds within the analysed model</td><td>√The contract holds throughout execution</td></tr><tr><td>Does each requested call carry a separately checkable proof?</td><td>X</td><td>X</td><td>X The shield itself may be verified</td><td>√The proof refers to the contract, tool and published information</td></tr><tr><td>Can proving task completion trigger the agreed settlement?</td><td>X</td><td>X</td><td>X</td><td>√An authorized request proving completion triggers settlement without further client</td></tr></table>

In PC2, the author specifies what must remain true of the state, through C, together with models D of the tools’ possible efects. These determine the obligation for each requested call, accounting for hidden state and intervening control steps (Definitions 3 and 4). The agent must prove this obligation for the call it has chosen. No separate permission policy needs to be designed and proved to preserve C for every call it permits. Each call’s proof establishes enabledness and preservation at all compliant active states compatible with the view and all active states admitted by the control window, covering every outcome allowed by the tool model. From these individual justifications and automatic control, Thm. 1 establishes compliance throughout execution once the contract has started. The agent can change its strategy without needing to verify that strategy. Coverage, fidelity and adequacy connect the argument to the executing routines and world safety (Sec. 4 and 5).

Producing proofs and checking them. The distinction between finding a proof and checking one underlies proof-carrying code (Necula, 1997). Finding an argument may require substantial search, auxiliary lemmas and problem-specific reasoning. A supplied proof records the steps that justify the conclusion, allowing a checker to verify them against the logical rules without discovering the argument itself. In PC2, this places the work of justifying a call with the agent requesting it, while the Airlock checks the result. Enforcement therefore does not require a general algorithm that decides whether each obligation is true. Constraints and tool models can give rise to undecidable obligations without preventing the Airlock from checking proofs of particular instances.

This separation allows the reasoning used to justify calls to become more sophisticated without changing the criterion for accepting them. Routine obligations can use existing automation and reusable lemmas, while more dificult ones can require additional search or specialist provers (Sec. 8.2). An accepted proof establishes the required preservation condition regardless of how it was found. If the agent cannot supply one, the requested call does not execute. Task completion may then fail, while checked calls and automatic control continue to provide the safety guarantee under the stated assumptions.

Defenses for tool-using agents. Prompt injection (Perez and Ribeiro, 2022; Greshake et al., 2023) allows content encountered during a task to steer an agent away from its instructions. AgentDojo (Debenedetti et al., 2024) provides an environment for evaluating such attacks and their defenses on realistic tasks. Deterministic defenses instead restrict execution independently of whether the model follows its instructions. CaMeL (Debenedetti et al., 2025) separates privileged planning from untrusted data and enforces capabilities on tool use. Progent (Shi et al., 2025) enforces programmable privilege policies, while FIDES (Costa et al., 2025) and RTBAS (Zhong et al., 2025) use information-flow restrictions. PC2 shares the principle that an adversarial model must remain subject to the execution mechanism. Its obligations address what the permitted call can do: the proof must cover every modelled outcome from the compliant active states compatible with the published view, including the changes allowed before execution.

Formal policies and contracts for agents. FORGE (Palumbo et al., 2026) enforces Datalog authorization rules over execution histories and externally supplied facts. Its enforcement argument relies on complete mediation and faithful policy-relevant observations. The guarantee concerns enforcement of the supplied policy, leaving its relationship to state safety to a separate argument.

Alignment Contracts (David et al., 2026) give a decidable contract language for permitted efects, budgets and disclosures. Their mechanized soundness theorem establishes that the event trace produced by a monitor satisfies the contract, independently of the agent’s policy. Applying the theorem requires correct event extraction and complete mediation, which the work explicitly distinguishes from its mechanized results. It also separates formal enforcement from heuristic mitigation of attacks involving the contents of permitted messages. The guarantee concerns the events represented in the contract language, with unmodelled disclosures and payload semantics outside its scope.

When authorization rules are supplied separately from a desired state constraint, their author must establish that every permitted call preserves that constraint. A policy can be too restrictive, rejecting safe calls, or insuficiently restrictive, allowing violations despite being enforced correctly. PC2 asks the contract author to specify what must remain true of the state, through C, together with models of the tools’ possible efects. From these objects, the published view and the allowed control steps, it derives the obligation for each requested call (Definitions 3 and 4). The proof must establish enabledness and preservation of C at every compliant active state compatible with the view and every active state admitted by its control window. The agent can therefore choose any strategy for carrying out the task, provided its calls carry proofs of these obligations. No separate permission policy needs to be designed and justified: each accepted proof establishes the preservation condition itself. This condition is exact relative to the chosen models and the states admitted by the view and control window: every rule that safely admits calls on the basis of the view implies it (Proposition A.1). Conservative models and dificulties in finding proofs can still prevent safe calls from executing (Sec. 4.1 and 8.2).

Proof-carrying provides a further form of generality. The Airlock checks a supplied proof of the constructed obligation, so its enforcement does not require a decision procedure for the obligation itself. The contract’s constraints and tool models can therefore give rise to undecidable obligations while retaining an independently checkable justification for every authorized call. Under the assumptions of Thm. 1, these checks and the automatic control steps preserve C throughout execution once the contract has started. Fidelity and adequacy connect this execution guarantee to world safety in Sec. 5, and Sec. 6.1 and 6.5 develop reusable module guarantees and the evidence and responsibilities supporting their application. Goalgated settlement further allows parties to agree on a completion procedure whose authorized execution requires no subsequent client approval.

Proofs, evidence and resource contracts. Type-Checked Compliance (Rashie and Rashi, 2026) proposes translating financial policies into Lean and checking proposed actions before invoking an API. Like PC2, it places a formal check between the proposed action and its execution. It does not develop the transition-based argument relating accepted obligations to all possible intermediate states and then to safety in the world. Proof-Carrying Agent Actions (Wang, 2026) uses “proof” for structured evidence attached to an action, including approval receipts, recorded assumptions and material for replay. Such evidence supports accountability, but is diferent from a checked derivation of an obligation over possible tool outcomes.

Agent Behavioral Contracts (Bhardwaj, 2026) develops a probabilistic account of compliance, drift and recovery, with composition results under assumptions on these processes. Resourcebounded Agent Contracts (Ye and Tan, 2026) organizes budgets, deadlines and delegation, and explicitly distinguishes stopping future calls from preventing an overrun within a call. PC2’s execution guarantee includes the states reached within each call and during control, rather than only the conditions checked when a new call is requested. Under its stated assumptions, preservation of the contract holds at every instant for arbitrary requests and proposed proofs.

Runtime enforcement and shields. Execution monitoring studies which properties can be enforced by observing a program and stopping unacceptable behaviour. Schneider’s characterization (Schneider, 2000) identifies safety as necessary for this form of enforcement: violations must be witnessed by an execution prefix. Shield synthesis (Bloem et al., 2015) and shielded reinforcement learning (Alshiekh et al., 2018) construct a component that prevents unsafe actions independently of the agent’s policy. Shielding has also been developed for continuous state and action spaces (Kim et al., 2024). In PC2, the Airlock checks a supplied proof rather than deciding the truth of the obligation by a general safety-analysis procedure. This permits predicates and state spaces beyond those supported by a particular synthesis algorithm. Relative to the chosen tool models, the obligation asks exactly for admissibility throughout the control window of every compliant active state with the published view: like the maximally permissive supervisor of supervisory control (Ramadge and Wonham, 1987), it restricts the agent only where the contract requires (Proposition A.1). It imposes no further restriction on how the agent chooses its actions or constructs its proof. Obtaining that proof is the task discussed in Sec. 8.2.

Proof-carrying code and authentication. Proof-carrying code (Necula, 1997) places proof construction with an untrusted code producer and proof checking with the consumer. Proofcarrying authentication (Appel and Felten, 1999) applies the same division to requests for access under a logical policy. PC2 adopts this established separation of responsibilities for tool calls. Its certificate concerns the efects of a particular requested operation, including nondeterministic outcomes and facts hidden by the view, while delayed obligations keep the certificate applicable as control runs. This makes the proof gate part of a continuing interaction with a changing system. The acting model and the prover can be replaced without changing the logical criterion for accepting a request.

Reasoning transparency. Inspecting an agent’s reasoning can help identify deception and unexpected behaviour, but the usefulness of this evidence depends on how much of the reasoning is visible and faithfully reported (Shah and Dragan, 2026). PC2 asks for a diferent kind of evidence: a proof that the requested call satisfies its obligation. That proof need not describe how the agent chose the call, and it can be produced by a separate prover. Its validity therefore does not depend on the transparency of the acting model. Monitoring can still help identify risks or omissions in the contract, while enforcement of its formal constraints rests on the proof and execution assumptions.

From verification to world safety. A proof about a mathematical description must be connected to the routines and services that execute it. Refinement mappings (Abadi and Lamport, 1991) provide a foundation for relating descriptions at diferent levels of abstraction. Verificationguided development and diferential testing, as used for Cedar (Disselkoen et al., 2024), can establish properties of a model and test whether an implementation agrees with it. Smith (Smith, 1985) examines the role of these interpretations in claims about real systems, and empirical studies of verified systems locate failures in specifications and interactions with unverified components (Fonseca et al., 2017). PC2 makes the relevant connections explicit through coverage, fidelity and adequacy. This separates the proof that execution preserves C from the evidence that compliant states describe safe situations in the world.

The structure follows the satisfaction argument of requirements engineering: properties of the environment together with a system specification must imply the intended requirement (Zave and Jackson, 1997). Assume–guarantee contracts (Benveniste et al., 2018) explain how a component’s guarantees depend on its environment and can support composition. In PC2, capability modules organize such reusable guarantees around tool models and the correspondence between states and the world. Assurance cases then organize the claims and evidence supporting an application of the theorem. They distinguish the argument for safety from confidence in its premises (Hawkins et al., 2011), and examine objections that could defeat that argument (Bloomfield and Rushby, 2020). Sec. 6.5 uses this perspective to make the dependencies of PC2’s guarantees visible, before Sec. 8.3 considers responsibility for what remains assumed.

Measurements and agreed meaning. Blockchain oracles face a related problem when external facts are used by an executable contract (Eskandari et al., 2021). Attestation and authenticated access can establish the source and processing of data (Zhang et al., 2016, 2020), while its interpretation still depends on what that source represents. Similarly, the Airlock’s record of an expert verdict establishes which procedure supplied an answer, without by itself establishing the answer’s accuracy. Goodhart’s law (Manheim and Garrabrant, 2018) and incomplete-contract accounts of AI alignment (Hadfield-Menell and Hadfield, 2019) explain why an optimized measure or a formal agreement may miss the intended objective. These concerns do not preclude proving adequacy for specified � and Safe. They motivate examining whether those choices express the intended protection. The distinction between engineering a system to satisfy a model and assessing a model against an existing system (Lee, 2016) also informs the diferent conditions attached to custody, formal interfaces and external behaviour in Sec. 6.1.

Guaranteed safe AI. The guaranteed safe AI framework (Dalrymple et al., 2024) combines a world model, a safety specification and a verifier. PC2 develops this approach for interactions mediated by tools: tool relations describe the relevant possible changes, the contract specifies constraints, and proofs are checked before requested operations run. The formalization can concentrate on the quantities needed by the contract and can use guarantees supplied by capability modules. It does not require a model of the agent’s internal reasoning. The coverage and fidelity assumptions nevertheless include every relevant change during execution, including outside changes, which is why restricting access and specifying external behaviour matter.

Agents, code and theorem proving. ReAct (Yao et al., 2023) and Toolformer (Schick et al., 2023) develop agents that combine reasoning with tool use, while CodeAct (Wang et al., 2024) uses code as the action interface. PC2 retains the ability to choose actions and adapt to observations, adding an obligation to requests that afect the protected system. LeanDojo (Yang et al., 2023), learned provers (Ren et al., 2025; Lin et al., 2025; Wang et al., 2025) and autoformalization (Wu et al., 2022) supply relevant methods for constructing proofs and specifications. Their application to the obligations and agreements considered here is the practical question developed in Sec. 8.

Commitment and cooperation. Program equilibrium (Tennenholtz, 2004) studies how commitments expressed through programs can support cooperation. PC2 provides a mechanism for enforcing agreed constraints on an agent’s interactions and for invoking authorized settlement without a further client veto. This addresses one part of the principal–agent problem (Jensen and Meckling, 1976), in which a principal delegates work without being able to rely on the delegate’s incentives. It also connects to the risks and opportunities of agents acting for diferent principals (Hammond et al., 2025). Choreographic and negotiation languages (Gopinathan et al., 2026; Wyse et al., 2026) study how interactions and agreements are organized. PC2 concerns their subsequent enforcement, provided the agreement has been expressed as a contract with the required guarantees.

Confidential computing. Confidential virtual machines, including AMD SEV-SNP (Advanced Micro Devices, 2020), provide mechanisms for protecting execution and attesting a running configuration to other parties. Work on confidential LLM inference (Chrapek et al., 2025) studies their use and cost for the models involved. These mechanisms can support the Airlock’s host and custody requirements and allow parties to identify the code to which they entrust an agreement. Their concrete configuration must provide the isolation and other execution properties identified in Sec. 5.2 and 5.3.

## 10 Conclusion

PC2 binds an AI agent to a contract it cannot break. The agent keeps its usual loop, and each of its tool calls carries a proof, checked before the call runs. Under the stated assumptions, from the moment the contract starts, the clauses hold at every instant for arbitrary requests and proposed proofs (Thm. 1). Since the Airlock isolates the agent, none of these assumptions depends on it: the checker, the Airlock, the contract and the tools are assessed once, and the guarantees hold for every agent, adversarial or not (Thm. 5). For the standard contract, settlement requires proof of the goal, an authorized settlement request requires no further client approval, and the contract otherwise ends as agreed (Thm. 2). What remains is named. Whether the guarantees reach the world depends on the fidelity of the tools and on the adequacy of the contract, for which we identify concrete deployment conditions (Sec. 5 and 6) and discuss responsibility for remaining assumptions (Sec. 8.3). Whether PC2 is useful depends on whether contracts can be written and proofs found at acceptable cost, which formalizer and prover agents address and a benchmark can measure (Sec. 8). PC2 guarantees safety under its assumptions, and termination by a deadline that the contract states as a clause is part of it (Thm. 5). Other forms of progress, such as settlement once the goal is reached, require additional conditions (Sec. 5.3). This paper treats one acting agent; several agents acting on shared resources, and subcontracts whose obligations survive at any depth, are the subject of a second paper.

## References

M. Abadi and L. Lamport. The existence of refinement mappings. Theoretical Computer Science, 82(2): 253–284, 1991. cited on pp. 23, 25, 38, and 50

M. Abadi and L. Lamport. Conjoining specifications. ACM Transactions on Programming Languages and Systems, 17(3):507–534, 1995. cited on p. 32

Advanced Micro Devices. AMD SEV-SNP: Strengthening VM isolation with integrity protection and more. White paper, 2020. cited on pp. 4 and 51

A. Agache, M. Brooker, A. Florescu, A. Iordache, A. Liguori, R. Neugebauer, P. Piwonka, and D.-M. Popa. Firecracker: Lightweight virtualization for serverless applications. In Proceedings of the 17th USENIX Symposium on Networked Systems Design and Implementation (NSDI), pages 419–434, 2020. cited on p. 27

B. Alpern and F. B. Schneider. Defining liveness. Information Processing Letters, 21(4):181–185, 1985. doi:10.1016/0020-0190(85)90056-0. cited on p. 28

M. Alshiekh, R. Bloem, R. Ehlers, B. Könighofer, S. Niekum, and U. Topcu. Safe reinforcement learning via shielding. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018. cited on pp. 47 and 49

A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada. Control barrier functions: Theory and applications. In 18th European Control Conference (ECC), pages 3420–3431, 2019. doi:10.23919/ECC.2019.8796030. cited on p. 38

J. P. Anderson. Computer security technology planning study. Technical Report ESD-TR-73-51, U.S. Air Force, Electronic Systems Division, 1972. cited on p. 26

Anthropic. Model Context Protocol, 2024. https://modelcontextprotocol.io. Specification maintained by the Agentic AI Foundation (Linux Foundation). cited on pp. 7 and 8

A. W. Appel and E. W. Felten. Proof-carrying authentication. In Proceedings of the 6th ACM Conference on Computer and Communications Security (CCS), pages 52–62, 1999. cited on p. 50

Y. Bai et al. Constitutional AI: Harmlessness from AI feedback. arXiv:2212.08073, 2022. cited on p. 2

A. Benveniste, B. Caillaud, D. Nickovic, R. Passerone, J.-B. Raclet, P. Reinkemeier, A. Sangiovanni-Vincentelli, W. Damm, T. A. Henzinger, and K. G. Larsen. Contracts for system design. Foundations and Trends in Electronic Design Automation, 12(2–3):124–400, 2018. doi:10.1561/1000000053. cited on pp. 42 and 50

V. P. Bhardwaj. Agent behavioral contracts: Formal specification and runtime enforcement for reliable autonomous AI agents. arXiv:2602.22302, 2026. cited on p. 49

F. Blanchini. Set invariance in control. Automatica, 35(11):1747–1767, 1999. doi:10.1016/S0005-1098(99)00113-2. cited on p. 38

R. Bloem, B. Könighofer, R. Könighofer, and C. Wang. Shield synthesis: Runtime enforcement for reactive systems. In Tools and Algorithms for the Construction and Analysis of Systems (TACAS), 2015. cited on pp. 47 and 49

R. Bloomfield and J. Rushby. Assurance 2.0: A manifesto. arXiv:2004.10474, 2020. cited on pp. 42 and 50

J. Bornholt, R. Joshi, V. Astrauskas, B. Cully, B. Kragl, S. Markle, K. Sauri, D. Schleit, G. Slatton, S. Tasiran, J. V. Gefen, and A. Warfield. Using lightweight formal methods to validate a key-value storage node in Amazon S3. In Proceedings of the 28th ACM Symposium on Operating Systems Principles (SOSP), pages 836–850, 2021. doi:10.1145/3477132.3483540. cited on p. 36

M. Chrapek, M. Copik, E. Mettaz, and T. Hoefler. Confidential LLM inference: Performance and cost across CPU and GPU TEEs. In IEEE International Symposium on Workload Characterization (IISWC), pages 84–98, 2025. https://arxiv.org/abs/2509.18886. cited on pp. 4, 45, and 51

M. Costa, B. Köpf, A. Kolluri, A. Paverd, M. Russinovich, A. Salem, S. Tople, L. Wutschitz, and S. Zanella-Béguelin. Securing AI agents with information-flow control. arXiv:2505.23643, 2025. cited on p. 48

D. Dalrymple, J. Skalse, Y. Bengio, S. Russell, M. Tegmark, S. Seshia, S. Omohundro, C. Szegedy, B. Goldhaber, N. Ammann, A. Abate, J. Halpern, C. Barrett, D. Zhao, T. Zhi-Xuan, J. Wing, and J. Tenenbaum. Towards guaranteed safe AI: A framework for ensuring robust and reliable AI systems. arXiv:2405.06624, 2024. cited on p. 51

I. David, M. Guarnieri, and A. Gervais. Alignment contracts for agentic security systems. arXiv:2605.00081, 2026. cited on p. 48

L. de Alfaro and T. A. Henzinger. Interface automata. In Proceedings of the 8th European Software Engineering Conference and 9th ACM SIGSOFT Symposium on the Foundations of Software Engineering (ESEC/FSE), pages 109–120, 2001. cited on p. 42

L. de Moura and S. Ullrich. The Lean 4 theorem prover and programming language. In Automated Deduction – CADE 28, volume 12699 of Lecture Notes in Computer Science, pages 625–635, 2021. cited on pp. 8, 14, 18, and 44

E. Debenedetti, J. Zhang, M. Balunović, L. Beurer-Kellner, M. Fischer, and F. Tramèr. AgentDojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2024. cited on pp. 2, 47, and 48

E. Debenedetti, I. Shumailov, T. Fan, J. Hayes, N. Carlini, D. Fabian, C. Kern, C. Shi, A. Terzis, and F. Tramèr. Defeating prompt injections by design. arXiv:2503.18813, 2025. cited on pp. 2, 47, and 48

J. B. Dennis and E. C. Van Horn. Programming semantics for multiprogrammed computations. Communications of the ACM, 9(3):143–155, 1966. cited on p. 35

E. W. Dijkstra. Guarded commands, nondeterminacy and formal derivation of programs. Communications of the ACM, 18(8):453–457, 1975. cited on pp. 11, 38, and 59

C. Disselkoen, A. Eline, S. He, K. Headley, M. Hicks, K. Hietala, J. Kastner, A. Mamat, M. McCutchen, N. Rungta, B. Shah, E. Torlak, and A. Wells. How we built Cedar: A verification-guided approach. In Companion Proceedings of the 32nd ACM International Conference on the Foundations of Software Engineering (FSE Companion), pages 351–357, 2024. cited on pp. 36 and 50

S. Eskandari, M. Salehi, W. C. Gu, and J. Clark. SoK: Oracles from the ground truth to market manipulation. In Proceedings of the 3rd ACM Conference on Advances in Financial Technologies (AFT ’21), pages 127–141, 2021. doi:10.1145/3479722.3480994. cited on pp. 50 and 67

R. Fagin, J. Y. Halpern, Y. Moses, and M. Y. Vardi. Reasoning About Knowledge. MIT Press, 1995. cited on p. 59

P. Fonseca, K. Zhang, X. Wang, and A. Krishnamurthy. An empirical study on the correctness of formally verified distributed systems. In Proceedings of the Twelfth European Conference on Computer Systems (EuroSys ’17), pages 328–343, 2017. doi:10.1145/3064176.3064183. cited on pp. 43 and 50

I. Gabriel and A. Kasirzadeh. The case for global benefit from AI. DeepMind Institute, Sept. 2026. https://institute.deepmind.com/essays/the-case-for-global-benefit-from-ai/. cited on p. 46

J. A. Goguen and J. Meseguer. Security policies and security models. In IEEE Symposium on Security and Privacy, pages 11–20, 1982. cited on p. 27

J. A. Goguen and J. Meseguer. Unwinding and inference control. In Proceedings of the 1984 IEEE Symposium on Security and Privacy, pages 75–86. IEEE Computer Society, 1984. cited on p. 69

I. Goldberg, D. Wagner, R. Thomas, and E. A. Brewer. A secure environment for untrusted helper applications: Confining the wily hacker. In Proceedings of the 6th USENIX Security Symposium, 1996. cited on p. 27

Google. Agent Payments Protocol (AP2) Specification, v0.2, 2025. https://ap2-protocol.org. cited on pp. 7 and 65

K. Gopinathan, J. Feser, M. Naim, Z. Tavares, and E. Bingham. Pact: A choreographic language for agentic ecosystems. arXiv:2605.03143, 2026. cited on pp. 45 and 51

C. G. Gray and D. R. Cheriton. Leases: An eficient fault-tolerant mechanism for distributed file cache consistency. In Proceedings of the 12th ACM Symposium on Operating Systems Principles (SOSP), pages 202–210, 1989. doi:10.1145/74850.74870. cited on pp. 28, 38, and 67

K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz. Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. In Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec), 2023. cited on pp. 2 and 48

R. Gu, J. Koenig, T. Ramananandro, Z. Shao, X. N. Wu, S.-C. Weng, H. Zhang, and Y. Guo. Deep specifications and certified abstraction layers. In ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL), pages 595–608, 2015. cited on p. 42

D. Hadfield-Menell and G. K. Hadfield. Incomplete contracting and AI alignment. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society (AIES), pages 417–422, 2019. doi:10.1145/3306618.3314250. cited on pp. 3, 24, 44, and 50

L. Hammond, A. Chan, J. Clifton, J. Hoelscher-Obermaier, A. Khan, E. McLean, C. Smith, W. Barfuss, J. Foerster, T. Gavenčiak, T. A. Han, E. Hughes, V. Kovařík, J. Kulveit, J. Z. Leibo, C. Oesterheld, C. S. de Witt, N. Shah, M. Wellman, P. Bova, T. Cimpeanu, C. Ezell, Q. Feuillade-Montixi, M. Franklin, E. Kran, I. Krawczuk, M. Lamparth, N. Laufer, A. Meinke, S. Motwani, A. Reuel, V. Conitzer, M. Dennis, I. Gabriel, A. Gleave, G. Hadfield, N. Haghtalab, A. Kasirzadeh, S. Krier, K. Larson, J. Lehman, D. C. Parkes, G. Piliouras, and I. Rahwan. Multi-agent risks from advanced AI. arXiv:2502.14143, 2025. cited on pp. 2, 46, and 51

R. Hawkins, T. Kelly, J. Knight, and P. Graydon. A new approach to creating clear safety arguments. In Advances in Systems Safety: Proceedings of the Nineteenth Safety-Critical Systems Symposium, pages 3–23. Springer, 2011. doi:10.1007/978-0-85729-133-2\_1. cited on pp. 43 and 50

C. A. R. Hoare. Proof of correctness of data representations. Acta Informatica, 1(4):271–281, 1972. doi:10.1007/BF00289507. cited on p. 23

M. Jackson. The world and the machine. In Proceedings of the 17th International Conference on Software Engineering (ICSE), pages 283–292. ACM, 1995. doi:10.1145/225014.225041. cited on pp. 22 and 37

M. Jackson. Problem Frames: Analysing and Structuring Software Development Problems. Addison-Wesley, 2001. cited on p. 23

J. Jacobs and A. Imas. Economic policy for AGI. DeepMind Institute, Sept. 2026. https://institute.deepmind.com/essays/economic-policy-for-agi/. cited on pp. 3 and 46

M. C. Jensen and W. H. Meckling. Theory of the firm: Managerial behavior, agency costs and ownership structure. Journal of Financial Economics, 3(4):305–360, 1976. cited on p. 51

C. B. Jones. Tentative steps toward a development method for interfering programs. ACM Transactions on Programming Languages and Systems, 5(4):596–619, 1983. doi:10.1145/69575.69577. cited on pp. 32, 38, and 62

R. Jung, D. Swasey, F. Sieczkowski, K. Svendsen, A. Turon, L. Birkedal, and D. Dreyer. Iris: Monoids and invariants as an orthogonal basis for concurrent reasoning. In Proceedings of the 42nd ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL), pages 637–650, 2015. doi:10.1145/2676726.2676980. cited on p. 42

T. Kelly and R. Weaver. The goal structuring notation: A safety argument notation. In Proceedings of the Dependable Systems and Networks Workshop on Assurance Cases, 2004. cited on p. 42

K. Kim, D. Corsi, A. Rodriguez, J. Lanier, B. Parellada, P. Baldi, C. Sanchez, and R. Fox. Realizable continuous-space shields for safe reinforcement learning. arXiv:2410.02038, 2024. cited on p. 49

Lean FRO. comparator: a trustworthy judge for Lean proofs, 2025a. https://github.com/leanprover/comparator. cited on p. 61

Lean FRO. lean4export: an exporter of Lean 4 declarations, 2025b. https://github.com/leanprover/lean4export. cited on p. 61

Lean FRO. Lean 4.35.0 release notes, 2026. https://lean-lang.org/doc/reference/latest/releases/v4.35.0/. cited on p. 62

E. A. Lee. Fundamental limits of cyber-physical systems modeling. ACM Transactions on Cyber-Physical Systems, 1(1):3:1–3:26, 2016. doi:10.1145/2912149. cited on pp. 37 and 50

Y. Lin, S. Tang, B. Lyu, Z. Yang, J.-H. Chung, H. Zhao, L. Jiang, Y. Geng, J. Ge, J. Sun, J. Wu, J. Gesi, X. Lu, D. Acuna, K. Yang, H. Lin, Y. Choi, D. Chen, S. Arora, and C. Jin. Goedel-Prover-V2: Scaling formal theorem proving with scafolded data synthesis and self-correction. arXiv:2508.03613, 2025. cited on pp. 3, 8, 45, and 51

N. Lynch and F. Vaandrager. Forward and backward simulations, Part I: Untimed systems. Information and Computation, 121(2):214–233, 1995. doi:10.1006/inco.1995.1134. cited on p. 24

D. Manheim and S. Garrabrant. Categorizing variants of Goodhart’s law. arXiv:1803.04585, 2018. cited on pp. 24, 44, and 50

M. S. Miller. Robust Composition: Towards a Unified Approach to Access Control and Concurrency Control. PhD thesis, Johns Hopkins University, 2006. cited on p. 35

C. Mohan, D. Haderle, B. Lindsay, H. Pirahesh, and P. Schwarz. ARIES: A transaction recovery method supporting fine-granularity locking and partial rollbacks using write-ahead logging. ACM Transactions on Database Systems, 17(1):94–162, 1992. doi:10.1145/128765.128770. cited on p. 66

G. C. Necula. Proof-carrying code. In Proceedings of the 24th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL), pages 106–119, 1997. cited on pp. 2, 48, 49, and 59

T. Nipkow, L. C. Paulson, and M. Wenzel. Isabelle/HOL: A ProofAssistant for Higher-Order Logic, volume 2283 of Lecture Notes in Computer Science. Springer, 2002. doi:10.1007/3-540-45949-9. https://isabelle.in.tum.de/doc/tutorial.pdf. cited on p. 8

D. C. North. Institutions, Institutional Change and Economic Performance. Cambridge University Press, 1990. cited on p. 2

P. O’Hearn, J. Reynolds, and H. Yang. Local reasoning about programs that alter data structures. In Computer Science Logic (CSL), volume 2142 of Lecture Notes in Computer Science, pages 1–19. Springer, 2001. doi:10.1007/3-540-44802-0\_1. cited on p. 42

P. E. O’Neil. The escrow transactional method. ACM Transactions on Database Systems, 11(4):405–430, 1986. doi:10.1145/7239.7265. cited on p. 66

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, et al. Training language models to follow instructions with human feedback. arXiv:2203.02155, 2022. cited on p. 2

D. Paglieri, L. Cross, T. Genewein, J. Z. Leibo, N. Tomašev, and A. S. Vezhnevets. A case study on emergent cheating and whistleblowing in autonomous research swarms. arXiv:2609.04170, 2026. cited on p. 61

N. Palumbo, S. Choudhary, J. Choi, G. Amir, P. Chalasani, I. Dillig, and S. Jha. Formal policy enforcement for agentic systems. arXiv:2602.16708, 2026. cited on pp. 2, 47, and 48

Z. Paraskevopoulou. Machine-generated, machine-checked proofs for a verified compiler (experience report). arXiv:2602.20082, 2026. https://arxiv.org/abs/2602.20082. cited on p. 8

F. Perez and I. Ribeiro. Ignore previous prompt: Attack techniques for language models. arXiv:2211.09527, 2022. cited on p. 48

R. Pollack. How to believe a machine-checked proof. In G. Sambin and J. M. Smith, editors, Twenty Five Years of Constructive Type Theory, pages 205–220. Oxford University Press, 1998. cited on p. 61

P. J. Ramadge and W. M. Wonham. Supervisory control of a class of discrete event processes. SIAM Journal on Control and Optimization, 25(1):206–230, 1987. cited on pp. 49 and 60

D. Rashie and V. Rashi. Type-checked compliance: Deterministic guardrails for agentic financial systems using Lean 4 theorem proving. arXiv:2604.01483, 2026. cited on p. 49

Z. Z. Ren, Z. Shao, J. Song, H. Xin, H. Wang, W. Zhao, L. Zhang, Z. Fu, Q. Zhu, D. Yang, Z. F. Wu, Z. Gou, S. Ma, H. Tang, Y. Liu, W. Gao, D. Guo, and C. Ruan. DeepSeek-Prover-V2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. arXiv:2504.21801, 2025. cited on pp. 3, 8, 45, and 51

J. C. Reynolds. Separation logic: A logic for shared mutable data structures. In IEEE Symposium on Logic in Computer Science (LICS), pages 55–74, 2002. cited on p. 42

J. Rushby. Noninterference, transitivity, and channel-control security policies. Technical Report CSL-92-02, SRI International, Computer Science Laboratory, 1992. cited on pp. 27 and 69

J. H. Saltzer and M. D. Schroeder. The protection of information in computer systems. Proceedings of the IEEE, 63(9):1278–1308, 1975. cited on pp. 6, 26, 35, and 38

T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, 2023. cited on pp. 2 and 51

F. B. Schneider. Enforceable security policies. ACM Transactions on Information and System Security, 3 (1):30–50, 2000. cited on pp. 28 and 49

L. Sha. Using simplicity to control complexity. IEEE Software, 18(4):20–28, 2001. doi:10.1109/MS.2001.936213. cited on p. 38

R. Shah and A. Dragan. The case for reasoning transparency. DeepMind Institute, Sept. 2026. https://institute.deepmind.com/essays/the-case-for-reasoning-transparency/. cited on p. 50

T. Shi, J. He, Z. Wang, H. Li, L. Wu, W. Guo, and D. Song. Progent: Securing AI agents with privilege control. arXiv:2504.11703, 2025. cited on pp. 2 and 48

B. C. Smith. The limits of correctness. ACM SIGCAS Computers and Society, 14–15(1–4):18–26, 1985. doi:10.1145/379486.379512. cited on pp. 22 and 50

M. Sozeau, S. Boulier, Y. Forster, N. Tabareau, and T. Winterhalter. Coq Coq correct! Verification of type checking and erasure for Coq, in Coq. Proceedings of the ACM on Programming Languages, 4(POPL): 8:1–8:28, 2020. cited on p. 62

M. Sozeau, Y. Forster, M. Lennon-Bertrand, J. B. Nielsen, N. Tabareau, and T. Winterhalter. Correct and complete type checking and certified erasure for Coq, in Coq. Journal of the ACM, 72(1), 2025. doi:10.1145/3706056. https://doi.org/10.1145/3706056. cited on p. 62

M. Tennenholtz. Program equilibrium. Games and Economic Behavior, 49(2):363–373, 2004. cited on p. 51

The Linux Foundation. Agent2Agent (A2A) Protocol Specification, v1.0.0, 2026. https://a2a-protocol.org. cited on p. 7

The mathlib Community. The Lean mathematical library. In Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs (CPP 2020), pages 367–381, 2020. doi:10.1145/3372885.3373824. cited on p. 8

The Rocq Development Team. The Rocq prover. Zenodo, n.d. https://doi.org/10.5281/zenodo.1003420. Concept DOI, all versions; accessed 16 October 2026. cited on p. 8

N. Tomašev, M. Franklin, J. Z. Leibo, J. Jacobs, W. A. Cunningham, I. Gabriel, and S. Osindero. Virtual agent economies. arXiv:2509.10147, 2025. cited on p. 3

J. Tretmans. Model based testing with labelled transition systems. In Formal Methods and Testing, volume 4949 of Lecture Notes in Computer Science, pages 1–38. Springer, 2008. doi:10.1007/978-3-540-78917-8\_1. cited on pp. 36 and 38

UNCITRAL. UNCITRAL Model Law on Automated Contracting with Guide to Enactment. United Nations, Vienna, 2025. https:

//uncitral.un.org/sites/uncitral.un.org/files/2424674e-mlautomatedcontracting-ebook.pdf. cited on p. 2

J. Viennot, G. Baudart, E. J. Gallego Arias, and M. Lelarge. MiniF2F in Rocq: Automatic translation between proof assistants—a case study. arXiv:2503.04763, 2025. https://arxiv.org/abs/2503.04763. cited on p. 62

H. Wang, M. Unsal, X. Lin, M. Baksys, J. Liu, M. D. Santos, F. Sung, M. Vinyes, Z. Ying, Z. Zhu, J. Lu, H. de Saxcé, B. Bailey, C. Song, C. Xiao, D. Zhang, E. Zhang, F. Pu, H. Zhu, J. Liu, J. Bayer, J. Michel, L. Yu, L. Dreyfus-Schmidt, L. Tunstall, L. Pagani, M. Machado, P. Bourigault, R. Wang, S. Polu, T. Barroyer, W.-D. Li, Y. Niu, Y. Fleureau, Y. Hu, Z. Yu, Z. Wang, Z. Yang, Z. Liu, and J. Li. Kimina-Prover preview: Towards large formal reasoning models with reinforcement learning. arXiv:2504.11354, 2025. cited on pp. 8, 45, and 51

X. Wang, Y. Chen, L. Yuan, Y. Zhang, Y. Li, H. Peng, and H. Ji. Executable code actions elicit better LLM agents. In International Conference on Machine Learning (ICML), 2024. cited on p. 51

Z. Wang. Proof-carrying agent actions: Model-agnostic runtime governance for heterogeneous agent systems. arXiv:2606.04104, 2026. cited on p. 49

F. Wiedijk. Pollack-inconsistency. Electronic Notes in Theoretical Computer Science, 285:85–100, 2012. cited on p. 61

Y. Wu, A. Q. Jiang, W. Li, M. N. Rabe, C. Staats, M. Jamnik, and C. Szegedy. Autoformalization with large language models. In Advances in Neural Information Processing Systems, 2022. cited on pp. 8, 44, and 51

T. Wyse, K. Bustos, Y. Volkova, and M. Kleiman-Weiner. Commitment to cooperation with self-negotiated contracts. arXiv:2607.22750, 2026. cited on pp. 45 and 51

K. Yang, A. M. Swope, A. Gu, R. Chalamala, P. Song, S. Yu, S. Godil, R. Prenger, and A. Anandkumar. LeanDojo: Theorem proving with retrieval-augmented language models. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2023. cited on pp. 8 and 51

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023. cited on pp. 2, 8, and 51

Q. Ye and J. Tan. Agent contracts: A formal framework for resource-bounded autonomous AI systems. arXiv:2601.08815, 2026. cited on p. 49

P. Zave and M. Jackson. Four dark corners of requirements engineering. ACM Transactions on Software Engineering and Methodology, 6(1):1–30, 1997. doi:10.1145/237432.237434. cited on pp. 22, 23, 37, and 50

F. Zhang, E. Cecchetti, K. Croman, A. Juels, and E. Shi. Town Crier: An authenticated data feed for smart contracts. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS), pages 270–282, 2016. doi:10.1145/2976749.2978326. cited on pp. 36, 50, and 67

F. Zhang, D. Maram, H. Malvai, S. Goldfeder, and A. Juels. DECO: Liberating web data using decentralized oracles for TLS. In Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security (CCS), pages 1919–1938, 2020. doi:10.1145/3372297.3417239. cited on pp. 36, 50, and 67

P. Y. Zhong, S. Chen, R. Wang, M. McCall, B. L. Titzer, H. Miller, and P. B. Gibbons. RTBAS: Defending LLM agents against prompt injection and privacy leakage. arXiv:2502.08966, 2025. cited on p. 48

## Appendices

The appendices provide implementation details, worked examples and further developments of the construction. They can be read independently alongside the relevant sections of the main text.

A The obligation as a weakest precondition. Why the obligation asks for exactly the safety of a call, and what this means for policy engines.

B Proof checking. The auxiliary program that checks requests and makes an accepted proof available for one tool invocation.

C Control design. A construction that maintains compliance by idling, correcting the state or terminating the contract, illustrated with a compute budget.

D Time between operations. How bounded gaps can be included in the operation models, with a corresponding convention for recorded observations.

E Correspondence in practice. Payment and booking examples show how state records support guarantees about safety and completion in the world.

F Failure without isolation. The ways in which actions outside the proof gate can invalidate coverage, fidelity or adequacy.

G A formal account of isolation. Conditions on concurrent steps and a proof that the agent introduces no efects beyond reference executions.

## A The obligation as a weakest precondition

Weakest preconditions. For a relation � on states and a predicate �, we use the following relational weakest precondition, which requires � to admit some outcome and all its outcomes to satisfy � (Dijkstra, 1975):

$$
\begin{array} { r } { \operatorname { w p } ( R , Q ) ( s ) : \Longleftrightarrow ( \exists s ^ { \prime } . R ( s , s ^ { \prime } ) ) \land \big ( \forall s ^ { \prime } . R ( s , s ^ { \prime } ) \implies Q ( s ^ { \prime } ) \big ) . } \end{array}
$$

The outcome is chosen against the requester, and the first conjunct excludes miracles: a relation with no outcome would otherwise establish any �. Admissibility is a weakest precondition, $\mathscr { A } _ { \sf T } ( i , \cdot ) = \mathrm { w p } ( \mathsf { D } _ { \sf T } ( i , \cdot , \cdot ) , \mathsf { C } )$ . Here an outcome of $\mathsf { D } _ { \mathsf { T } }$ is a state admitted during an invocation, rather than necessarily a state at which it returns. The existential condition therefore establishes enabledness in the model, not termination of exec , and the universal condition requires C throughout the call when coverage holds. In proof-carrying code, the producer proves a verification condition computed as the weakest precondition of its program for the consumer’s safety policy (Necula, 1997); in PC2, the agent proves that of one call.

What the agent knows. The agent does not know the state, only the view � on which its request is based, the contract, and that the state complies with it, which Theorem 1 makes true. For a call that may start after up to � control steps, the states it cannot rule out are

$$
K _ { n } ( \nu ) \ : = \ \left\{ s ^ { \prime } \in \Sigma _ { \mathrm { a c t } } \ : \ \exists s \in \Sigma _ { \mathrm { a c t } } . { \big \lor } \mathrm { i } \mathrm { e w } ( s ) = \nu \land \mathsf { C } ( s ) \land \mathsf { D } _ { \mathrm { c t r l } } ^ { \leqslant n } ( s , s ^ { \prime } ) \right\} ,
$$

and, by Definition 4,

$$
\mathrm { O b l } _ { \mathsf { T } } ^ { \leqslant n } ( i , \upsilon ) \iff \forall s ^ { \prime } \in K _ { n } ( \upsilon ) . \mathsf { w p } \big ( \mathsf { D } _ { \mathsf { T } } ( i , \cdot , \cdot ) , \mathsf { C } \big ) ( s ^ { \prime } ) .\tag{A.1}
$$

The obligation states that the agent knows the call to be admissible, in the sense of the logic of knowledge, where a fact is known when it holds at every state considered possible (Fagin et al., 1995).

The most permissive gate. A gate specifies, from a request and the view � on which it is based, whether the call may run: it is a predicate $G ( \mathsf { T } , i , n , \upsilon )$ . A gate is safe if every call it admits is admissible at every state from which it may start: $G ( \mathsf { T } , i , n , \upsilon )$ implies $\mathcal { A } _ { \sf T } ( i , s ^ { \prime } )$ for every $s ^ { \prime } \in K _ { n } ( \nu )$

Proposition A.1 The most permissive safe gate

The obligation is a safe gate, and every safe gate � satisfies $G ( \mathsf { T } , i , n , \upsilon ) \implies \mathsf { O b l } _ { \mathsf { T } } ^ { \leqslant n } ( i , \upsilon )$ for all ${ \sf T } , i ,$ � and �. In particular, if the obligations of a contract form an undecidable predicate, every decidable safe gate rejects some call whose obligation holds.

Proof. By $( \mathrm { A } . 1 ) , \mathrm { O b l } _ { \mathsf { T } } ^ { \leqslant n } ( i , \nu )$ holds exactly when $\mathcal { A } _ { \sf T } ( i , s ^ { \prime } )$ holds for every $s ^ { \prime } \in K _ { n } ( \nu )$ . This is what safety of the obligation requires, and what safety of � yields when $G ( \mathsf { T } , i , n , \nu )$ holds. A decidable safe gate that admitted every call whose obligation holds would coincide with the obligation, which would then be decidable. ■

The proposition is relative to the models and to the view: models that admit fewer outcomes while remaining enabled, views that show more and shorter windows all weaken the obligation, and a gate that also used earlier views could admit more calls, which a contract obtains by putting the relevant history in the state. It concerns one call, while the contract must hold for the rest of the execution. The two agree because the control step keeps C from every compliant active state ( A3 ): a call that keeps C leaves a state from which the contract can be kept, so a safe gate needs no look-ahead. In the terms of supervisory control, C is controllable, and the obligation is the maximally permissive supervisor, which excludes exactly the calls that fail enabledness or preservation at some state in $K _ { n } ( \nu )$ (Ramadge and Wonham, 1987).

Policy engines. A policy engine admits calls by a fixed rule that it decides itself: a decidable gate, whose author must show that it is safe. By Proposition A.1, a safe rule admits only calls whose obligation holds, and if its safety is proved, each call it admits comes with a proof of its obligation: evaluate the rule on the request, and apply the proof of safety. An agent constrained by a safe policy can therefore run under PC2, which admits, in addition, every other call whose obligation is proved: the agent may follow any strategy whose calls keep the contract. A fixed rule rejects safe calls wherever it is stricter than the obligation, as it must be wherever the obligation is undecidable. In the travel contract, a rule that caps each of three nights at 600 EUR keeps the budget of 2000 EUR, and rejects a first night at 900 EUR followed by two at 500 EUR, although these keep the budget too. The exact rule is the obligation itself: a policy engine can enforce it only where it is decidable, and only after someone has derived it from the contract, while PC2 derives it and checks a proof of it.

## B Details on the proof-checking program

Alg. B.1 gives the routines used by Algorithm 1 to check a request while control continues. The check concerns a particular tool, input and recorded view. Its correctness therefore requires both a sound checker, as in A1 , and the connection between the checked statement and the call required by A4(iii) .

Forming the obligation. For a request $( \mathsf { T } , i , k _ { 0 } , n , \pi )$ , the Airlock retrieves the unchanged view $\nu [ k _ { 0 } ]$ and checks that ${ \sf T } \in \mathcal T$ and $i \in I _ { \mathsf { T } }$ . It then forms the statement $\mathrm { O b l } _ { \mathsf { T } } ^ { \leqslant n } ( i , \upsilon [ k _ { 0 } ] )$ from the contract’s agreed definitions. The submitted proof must establish this statement under the permitted axioms. Checking a statement supplied by the requester would not sufice, since it might express a diferent condition.

This requirement also concerns how a statement is represented. Source text acquires its meaning from the notations and definitions in scope. For example, a macro that replaces the written obligation with True would make trivial a valid proof of the resulting statement, while establishing nothing about the requested call. Agents have exploited changes to notation in a Lean submission system in this way (Paglieri et al., 2026). The issue is the connection between the intended statement and the one checked, a distinction discussed by Pollack (1998); Wiedijk (2012).

Algorithm B.1 · Details on proof checking.   
1 shared variables � ← none; ℎ ← none   
2 def check $( \top , i , k _ { 0 } , n , \pi , k _ { \mathrm { l a s t } } , \nu )$ ⊲ runs in its own process   
3 � ← len(�) − 1 ⊲ index of the last published view   
4 if T ∈ T and $k _ { \mathrm { l a s t } } < k _ { 0 } \leq k \leq k _ { 0 } + n$   
⊲ Checks that � : � , so we can build Obl<sup>⩽�</sup> (�, �[�<sub>0</sub>])   
5 if type\_check(�, �<sub>T</sub>)   
⊲ Checks that � is a valid prooffor Obl<sup>⩽�</sup>(�, �[� ])   
6 if kernel\_check(�, Obl<sup>⩽�</sup>(�, �[�<sub>0</sub>]))   
7 � ← correct; return   
8 � ← incorrect   
9 def start\_proof\_check(T, �, �<sub>0</sub>, �, �, �<sub>last</sub>, �)   
10 if � = computing then kill\_process(ℎ)   
11 � ← computing ⊲ any earlier verdict is dropped   
12 ℎ ← new\_process(check, (T, �, �<sub>0</sub>, �, �, �<sub>last</sub>, �))   
13 def is\_proof\_correct()   
14 return � = correct   
15 def consume\_proof() ⊲ an accepted proof authorizes one call   
16 � ← none

In Lean, the obligation can instead be formed directly as an expression in the kernel’s language, using the fixed contract definitions and an encoding of the recorded view. That encoding must faithfully represent the view. A kernel expression does not depend on submitted notation or macros for its meaning. Lean’s export format records such expressions and their dependencies, and comparator checks a submitted solution against an independently supplied statement and permitted axioms (Lean FRO, 2025b,a). These facilities support the check, provided the statement against which the solution is compared is formed independently of untrusted source.

From a verdict to a call. Each check uses a fixed request and recorded view, and its verdict belongs only to that request. In the pseudocode, kill\_process cancels the previous check before its replacement starts, so the cancelled check cannot subsequently publish a verdict. Replacing a request also discards any earlier verdict. The execution loop checks that the current request remains within its control window before running it, and consume\_proof prevents its reuse.

The invocation must use the same tool and input that appear in the checked obligation. For an input containing executable code, this also requires the runtime behaviour of that code to agree with the definitions used in the proof, a requirement on the compiler and runtime. Running the checker in a separate process allows control to continue while a proof is checked. The checker must also be prevented from altering the contract, recorded views or execution of a running routine. Process separation supports these requirements, but must be accompanied by the access restrictions needed to satisfy A4 and A7 .

Independent checking. The same formal proof can be checked by several implementations. If acceptance requires all of them to accept the same statement under the agreed definitions and axioms, soundness of any one of them sufices for soundness of the combined check, provided their results are combined correctly. Lean provides export-based checking and, from version 4.35, an option to run several bundled checkers (Lean FRO, 2026). This ofers a way to reduce reliance on one implementation of the checking rules.

Rocq also has a separately implemented checker through the MetaRocq project. Its safe checker is written and verified in Rocq for PCUIC, a formal calculus underlying Rocq. Its correctness and completeness results concern that calculus under their stated assumptions, including strong normalization (Sozeau et al., 2020, 2025). Within this scope it provides another way to support A1 .

Using diferent proof assistants requires an additional connection between their statements. For example, recent work translates MiniF2F statements into Rocq, with human review of their semantic agreement (Viennot et al., 2025). To use such a translation for a PC2 obligation, one also needs a proof of the translated statement and a justification that it expresses the intended obligation. Rechecking the same formal statement with another kernel does not require this translation.

## C A possible design for the control ctrl

The control routine of Section 3.5.1 keeps a contract compliant while the agent prepares calls and their proofs are checked. Depending on the state it reads, it waits, corrects the state or closes the contract. We now explain how these choices can establish the certificate required by A3 , and illustrate the argument with a compute budget. The construction connects the clause to the interventions available to control: every compliant active state must leave at least one of these choices safe.

Establishing the certificate. Fix the idle period � and the region $C _ { \mathrm { i d l e } }$ of Section 3.5.1. Outside Act, a compliant state belongs to $C _ { \mathrm { i d l e } } .$ , so the model of one idle period admits only compliant states. In Act $\cap F i x ,$ the correcting routine must preserve � throughout its execution and return to $C _ { \mathrm { i d l e } }$ . In End, the termination routine must preserve � until the contract closes, including at closing. Each branch must also be enabled wherever it is selected. The model $\mathsf { D } _ { \mathsf { c t r } 1 }$ covers the whole control period, from its initial reading until it is ready to hand over to the next invocation. The branch guarantees must therefore include the outside changes and costs over that whole period. Together, these conditions give the controller line of A3 .

Initialization supplies its other role. From Ready, the initializer establishes � before making the contract active, as when it sets the status to ongoing at the end of initialization. From the other inactive states it leaves the contract inactive. Providing an enabled model for both cases gives the initializer requirements of A3 . Only the tests used by the routine, namely Ready, Act and $F i x ,$ need to be decidable. The clause and $C _ { \mathrm { i d l e } }$ can remain more general predicates, with their relationship to these tests established by the certificate.

This construction includes the simpler controllers discussed in Section 3.5.1. If outside changes preserve $C ,$ then $C _ { \mathrm { i d l e } } = C$ and control can simply wait, as in the stability condition of rely– guarantee reasoning (Jones, 1983). If no correction is supplied, control waits or terminates, as in the travel contract. When correction is available, returning well inside $C _ { \mathrm { i d l e } }$ leaves room for further activity before another intervention is needed.

When a control invocation from a compliant active state returns without closing the contract, the state is either in $C _ { \mathrm { i d l e } }$ after correction or in a state reached by one idle period from that region. This describes the states at the completion of control, as illustrated in Figure 4(a). A requested tool may subsequently move the state elsewhere within �. The delayed obligations in panel (b) use the full model $\mathsf { D } _ { \mathsf { c t r l } . }$ , which must also include the intermediate states of control invocations.

Example: a compute budget. A company lets an agent run cloud jobs within a budget $B ,$ with the contract to close by a deadline �. The state records the bill $b ,$ an upper bound � on the combined charging rate of the running jobs, the time � and the contract’s status ℓ. Outside changes can increase the bill at rate at most � or stop jobs, but only an authorized call can add jobs or increase the rate. In this example we model readings and branch selection as instantaneous, while stopping the jobs can take up to � units of time. The contract must reserve enough money and time for this shutdown.

To express that reserve throughout execution, the state also records a remaining shutdown allowance ℎ. While jobs run normally, $h = \delta .$ . Once shutdown begins, ℎ counts down with elapsed time, and the model requires the jobs to have stopped by the time it reaches zero. If no jobs remain, $r = h = 0$ . We use the clause

$$
{ \mathsf { C } } ( s ) \quad : \Longleftrightarrow \quad b + r h \leq B \quad \wedge \quad \big ( \ell = \mathsf { o n g o i n g } \implies t + h \leq d \big ) .
$$

The term �ℎ reserves the maximum further charge during shutdown. Thus the clause protects the budget while retaining the means to stop, and the agent’s calls cannot spend that reserve.

Control can idle for � whenever

$$
b + r ( p + h ) \leq B , \qquad t + p + h \leq d .
$$

These decidable tests certify membership in $C _ { \mathrm { i d l e } } i$ : throughout the idle period the bill can rise by at most $^ { r p , }$ while enough money and time remain for shutdown. Use the complement of this tested region for $A c t$ . When only the budget test fails, there is still time to correct the state. A simple fix stops the running jobs and leaves the contract active. When the time test fails, control stops the jobs and closes the contract instead.

Both interventions preserve the clause. After $0 \leq u \leq h$ time units during shutdown, its model gives $b ^ { \prime } \leq b + r u , r ^ { \prime } \leq r , h ^ { \prime } \leq h - u$ and $t ^ { \prime } = t + u ,$ , hence

$$
b ^ { \prime } + r ^ { \prime } h ^ { \prime } \leq b + r u + r ( h - u ) = b + r h \leq B , \qquad t ^ { \prime } + h ^ { \prime } \leq t + h \leq d .
$$

At completion, $r ^ { \prime } = h ^ { \prime } = 0$ . In the correcting branch, the time test held before the intervention, so $t ^ { \prime } + p \leq t + h + p \leq d . $ . The resulting state therefore belongs to $C _ { \mathrm { i d l e } }$ . In the terminating branch, shutdown finishes within the reserved time and the contract closes by �, with the bill still at most �. Initialization starts the contract only when the waiting tests have been established; otherwise it leaves it inactive.

The same accounting applies to a tool that launches more jobs: its obligation requires the clause, including the shutdown reserve, throughout the call. The call need not leave enough room to wait for another whole period, because control can intervene immediately afterwards. This distinguishes the protection supplied by the contract from the additional room for uninterrupted work supplied by $C _ { \mathrm { i d l e } }$

## D Time between operations

A4(i) takes the operations of the Airlock to be contiguous: each starts when the previous one ends. In an implementation, a routine may return before the host is ready to invoke the next one, and the system can continue to change during this interval. We account for a gap of at most $t _ { 0 }$ by including it in the preceding operation and enlarging the tool models accordingly.

Together with the sampling convention specified below, this lets us apply the results of Section 4 to the extended operations. Their models and control certificate must cover the added intervals.

Extended operations. Between the return of a routine and the invocation of the next one, no tool routine runs. The state can still change, by outside changes: time advances, prices move, mail arrives. Over a time of at most $t _ { 0 } ,$ these are the changes that the model of idle admits (Section 3): $\mathsf { D } _ { \mathsf { i d l e } } ( t _ { 0 } , z , s ^ { \prime \prime } )$ holds when outside changes can lead from � to $s ^ { \prime \prime }$ within a time of at most $t _ { 0 } .$ . In particular, $\mathsf { D } _ { \mathsf { i d l e } } ( t _ { 0 } , z , z )$ holds for every �, since that time may be zero.

For each tool ${ \mathsf { T } } \in { \mathcal { T } } \cup \{ { \mathsf { c t r } } { \mathsf { l } } \}$ , let ${ \mathsf { T } } ^ { \prime }$ be the corresponding extended tool, with the same input space. An operation of ${ \mathsf { T } } ^ { \prime }$ executes the routine exec and includes the subsequent gap. Its model is

$$
\mathsf { D } _ { \mathsf { T } ^ { \prime } } ( i , s , s ^ { \prime \prime } ) : \Longleftrightarrow \exists z . \mathsf { D } _ { \mathsf { T } } ( i , s , z ) \land \mathsf { D } _ { \mathsf { i d l e } } ( t _ { 0 } , z , s ^ { \prime \prime } ) .\tag{D.1}
$$

The model of ${ \sf T } ^ { \prime }$ admits every state that the model of $\top$ admits, taking $z = s ^ { \prime \prime }$ , and every state that outside changes can reach within $t _ { 0 }$ from such a state, in particular from the state in which the routine returns. It is enabled exactly where the model of T is. Let $\mathcal { K } ^ { \prime }$ be the contract $\mathcal { K }$ with the tools ${ \sf T } ^ { \prime } { } _ { ; }$ , for ${ \sf T } \in \mathcal { T } .$ , and the control tool $\cot \mathsf { r } \mathsf { l } ^ { \prime }$ . Its state space, clauses, view and inactivity test are those of $\mathcal { K }$

An operation of ${ \mathsf { T } } ^ { \prime }$ starts when the Airlock invokes exec<sub>T</sub>, as an operation of T does, but it ends only when the Airlock invokes the next routine, or when the contract closes. It consists of the execution of the routine followed by the time between operations. The underlying tool routine is unchanged, but its extended operation includes the time spent by the host before the next invocation. If the routines run on the intervals $[ \tau _ { 1 } , \tau _ { 1 } ^ { \prime } ] , [ \tau _ { 2 } , \tau _ { 2 } ^ { \prime } ] , . . .$ , with $\tau _ { k } ^ { \prime } \le \tau _ { k + 1 } \le \tau _ { k } ^ { \prime } + t _ { 0 } .$ , the operations of $\mathcal { K }$ are these intervals, while those of $\bar { \mathcal { K } } ^ { \prime }$ are $[ \tau _ { 1 } , \tau _ { 2 } ] , [ \tau _ { 2 } , \tau _ { 3 } ] , . . .$ . The extended operations are therefore contiguous, as required by A4(i) .

Readings at operation boundaries. To use Lemma 1, a recorded view must describe the state at a boundary between the operations counted by its window. We therefore use the following sampling convention for the extended operations: the reading associated with a completed operation is taken at the handof to the next routine, and views and the inactivity test use that reading. Sampling and dispatch are treated as a single boundary event in this abstraction. This convention, together with the bound $t _ { 0 } ,$ is a condition of the construction. It connects the readings to the extended operations as required by A4(ii) . Together with the exclusion of intervening requested calls in A4(iii) , this ensures that a call based on an earlier recorded view is separated from its reading only by complete operations of $\mathsf { c t r } \mathsf { l } _ { } ^ { \prime } ,$ , so the control-window argument applies.

Guarantees for the extended operations. With this sampling convention, we apply the definitions and results of Sections 3 and 4 $\mathrm { t o } \ \mathcal { K } ^ { \prime }$ . In A2 , $E _ { \mathsf { T } ^ { \prime } }$ now describes the states reached throughout the extended operation, including the gap. The Airlock forms the obligations of $\mathcal { K } ^ { \prime }$ from the models $\mathsf { D } _ { \mathsf { T } ^ { \prime } }$ , and their windows from $\mathsf { D } _ { \mathsf { c t r } \mathsf { l ^ { \prime } } }$ . Under A1 – A4 for $\mathcal { K } ^ { \prime }$ , Theorem 1 gives the clauses at every instant from the start, including the time between operations, which now lies within operations of $\mathcal { K } ^ { \prime }$ . Since $( \mathsf { T } | _ { \mathsf { P r e } } ) ^ { \prime } = \mathsf { T } ^ { \prime } | _ { \mathsf { P r e } }$ , the settlement tool of a standard contract remains goal-gated, and since outside changes do not modify the label, Theorem 2 applies too. For $t _ { 0 } = 0 , \mathsf { D } _ { \mathsf { i d l e } } ( 0 , \cdot , \cdot )$ is the identity, and $\mathcal { K } ^ { \prime }$ is $\mathcal { K } .$

These guarantees require the assumptions to hold for $\mathcal { K } ^ { \prime }$ . Coverage must account for the routine and the following gap, and the control certificate must establish initialization and compliance for the enlarged model $\mathsf { D } _ { \mathsf { c t r } \mathsf { L ^ { \prime } } }$ . For an idling branch of duration $p ,$ , this requires accounting for up to $p + t _ { 0 }$ of outside changes. The correcting and terminating routines must be checked against their enlarged models as well. The agent’s obligations also use the enlarged models, so an accepted proof establishes compliance during the call and its following gap.

To apply the world-safety result of Theorem 3, $\widehat { E } _ { \mathsf { T } ^ { \prime } }$ must likewise describe the world and state throughout the extended operation, and A5 must hold over that whole interval. Preserving the correspondence during the added gap then connects the compliance guarantee to safety in the world through A6 .

## E Two worked examples of correspondence

This appendix works through two correspondences of the kind studied in Section 5.1. In the payment ledger, the state bounds what happens in the world, as safety requires. In the booking confirmations, what the state records must be true of the world, as completion requires.

## E.1 A payment ledger

The structure needed is simple, and we describe it in words. The contract lets the agent pay merchants within a budget $B ,$ fees included, through a payment provider that issues payment instructions with a spending limit, such as single-use virtual cards, pre-authorizations, or payment mandates with a maximum amount (Google, 2025).

• The state contains a ledger kept by the Airlock: a list of entries, each with three fields, an identifier, a payee and a limit.

• The world contains the provider’s records for the account: for each instruction it has issued, its identifier, payee and limit, and the charges posted against it.

• The tool $\mathsf { p a y } ( m , q )$ pays the merchant � up to �. Its routine appends the entry $( j , m , q )$ to the ledger, under a fresh identifier $j ,$ and only then asks the provider to issue the instruction $( j , m , q )$ . A retry reuses $j ,$ which the provider treats as an idempotency key. The model of pay admits the ledger unchanged or with the new entry.

For an identifier �, write $r _ { j } ( s )$ for the limit of its entry in the ledger, $u _ { j } ( \boldsymbol { w } )$ for the limit the provider has issued under �, and $a _ { j } ( \nu )$ for the total charged under $j ,$ each zero if there is no such entry, instruction or charge. The clause bounds what the ledger has reserved, and safety bounds what has actually been charged:

$$
b ( s ) : = \sum _ { j } r _ { j } ( s ) , \qquad a ( w ) : = \sum _ { j } a _ { j } ( w ) , \qquad \mathsf { C } ( s ) : = b ( s ) \le B , \qquad \mathsf { S a f e } ( w ) : = a ( w ) \le B .\tag{E.1}
$$

The budget bound remains part of the clause at closing, so settlement or termination cannot discharge the obligation to respect it. With this model, admissibility of $\mathsf { p a y } ( m , q )$ at a compliant state reduces to $b ( s ) + q \leq B$ , as for the tools of Section 3.5.2. When the view reveals the reserved total, the ordinary obligation reduces to the same inequality. In the travel contract, $b ( s )$ is the amount spent in Section 2.2: it counts reservations, not receipts. The correspondence relates the provider’s records to the ledger:

$$
\rho ( w , s ) \quad \iff \quad \forall j . \quad a _ { j } ( w ) \leq u _ { j } ( w ) \leq r _ { j } ( s ) .\tag{E.2}
$$

Every issued limit is covered by an entry of the ledger, and every charge stays within its limit.

Two conditions maintain this correspondence. The reservation discipline: only the Airlock’s routines ask the provider to issue instructions, each after recording its entry; entries are never removed or lowered, issued limits never change, and retries reuse their identifier. The service condition: the provider lets the charges under an instruction, fees and retries included, reach at most its limit, and accepts no charge without an instruction. Both conditions apply at every instant, to all routines and outside interactions. Charges may arrive after a routine returns, so entries are retained to cover them.

## Proposition E.1 Preserving budget correspondence

The reservation discipline and service condition preserve (E.2) whenever it holds initially. In particular, they imply the first line of A5 .

Proof. For an existing instruction, $u _ { j } \leq r _ { j }$ persists because its issued limit is fixed and its reservation never decreases. A new instruction satisfies the inequality because reservation precedes issuance. Before issuance, $u _ { j } = 0 \leq r _ { j }$ . The service condition gives $a _ { j } \leq u _ { j }$ at every instant, so correspondence holds throughout, including between routines. ■

Summing (E.2) gives $a ( w ) ~ \leq ~ b ( s )$ , so the clause implies $\mathsf { S a f e } ( \boldsymbol { w } )$ , establishing the safety implication of A6 . A new entry with limit $q$ requires $b ( s ) + q \leq B$ at every invocation state admitted by the view and control window. Under A1 – A4 , Theorem 1 preserves this budget clause. Together with Proposition E.1, and a start in which the empty ledger describes the account, as for a fresh account with no issued instructions or charges ( A5 ), this yields the world-safety guarantee. Recording expenditure only when a receipt arrives would instead leave issued instructions temporarily uncovered. The ledger reserves a quantity before it can be consumed, as the escrow method of transaction processing does for bounded quantities $\mathrm { ( O N e i l }$ 1986), and records an action before performing it, as write-ahead logging does (Mohan et al., 1992).

The reservation discipline concerns the routine, which can be verified, and the custody of the provider’s credentials, which only the Airlock may use (Section 6.1). The service condition concerns the provider: it is a property of a service that the parties do not build, which the provider must support or the parties must accept. Where a provider allows charges to exceed a limit by a tolerance, the routine must record the limit plus that tolerance.

The guarantee can extend beyond closing. Suppose the hypotheses of Theorem 3 hold and the session closes at time $\tau _ { c }$ with state $s _ { c }$ . If no new instructions can be issued afterwards, existing issued limits remain fixed, and the service condition continues to hold, then for every $\tau \geq \tau _ { c : }$

$$
a ( w ( \tau ) ) \leq b ( s _ { c } ) \leq B .\tag{E.3}
$$

The reservations at closing already cover the charges that may still arrive. This protection therefore survives without the Airlock continuing to run, provided the restrictions on issuance and charging remain efective.

## E.2 Booking confirmations

The ledger construction controls which instructions can produce charges. A booking confirmation raises a diferent question: what does a service’s response establish, and for how long?

## Example 3 Booking confirmations

Let �(�) be a finite set of recorded confirmations $( b , d )$ , where � identifies a particular reservation and � is its expiry time. Write Issued(�, $b , d )$ when the world’s history contains the agreed service’s confirmation of this reservation until $d ,$ and Held $( \boldsymbol { w } , \boldsymbol { b } )$ when the hotel actually holds it. With $t _ { \mathcal { W } } ( \boldsymbol { w } )$ denoting time in the world, consider

$$
\begin{array} { r } { \begin{array} { r l } { \rho _ { \mathrm { b o o k } } ( w , s ) } & { ~ \Longleftrightarrow ~ \forall ( b , d ) \in R ( s ) . } \end{array} \begin{array} { r l } { | \mathord { \mathrm { s s u e d } } ( w , b , d ) \wedge \big ( t _ { \mathcal { W } } ( w ) < d \implies \mathsf { H e l d } ( w , b ) \big ) . } \end{array} } \end{array}\tag{E.4}
$$

The first conjunct states what the service reported. The second connects that report to a reservation still held before expiry. After expiry, the record continues to describe a past confirmation, but no longer asserts that the reservation is held.

The recording condition requires entries to be added only from authenticated confirmations of the agreed service, with the reservation and expiry recorded correctly. Existing entries may be retained or removed, and any replacement must meet the same condition. The reservation condition requires every confirmed reservation to be held until its stated expiry, including during other operations and between routines. This condition concerns both the service issuing the confirmation and the hotel holding the reservation. A confirmation with an expiry is a lease: a grant that holds until a stated time (Gray and Cheriton, 1989).

## Proposition E.2 Preserving booking correspondence

The recording and reservation conditions in Example 3 preserve (E.4) whenever it holds initially, provided the world’s history retains issued confirmations. They therefore establish the booking component of the first line $O f \left( { \underline { { A 5 } } } \right) .$

Proof. Consider any later instant and an entry $( b , d )$ still in the record. It was present initially or added from a correctly recorded confirmation. In either case, Issued $( w , b , d )$ holds, since history retains that event. If $t _ { \mathcal { W } } ( \boldsymbol { w } ) \ < \ d _ { : }$ , the reservation condition gives ${ \mathsf { H e l d } } ( w , b )$ . Both conjuncts therefore hold throughout execution, including between routines. ■

Authentication supports the recording condition, but does not establish the reservation condition. It establishes where a report came from, not that the report is true, a distinction familiar from authenticated data feeds and blockchain oracles (Zhang et al., 2016, 2020; Eskandari et al., 2021). The service may have sent an authentic confirmation even though the hotel later cancels the booking. Conversely, the reservation condition need not prevent cancellation after expiry. For a goal that requires the reservation still to be held, the contract must also establish that the confirmation has not expired when settlement begins. For instance, a further correspondence claim $t _ { \mathcal { W } } ( \boldsymbol { w } ) \leq t ( s )$ , where $t ( s )$ is a conservative time bound, makes $t ( s ) < d$ suficient. This bound must cover delays before invocation: an earlier clock reading alone does not provide it. Together with (E.4), it supplies the connection from a recorded confirmation to Held needed by Corollary 4.

For the travel contract, this gives the following goal in the world.

Example 4 Settlement of a travel contract

For the travel contract, let Nights be the requested nights, and let RoomHeld $( \boldsymbol { w } , \boldsymbol { n } )$ mean that a hotel holds a reservation for the traveller on night �. A possible world goal is

$$
G o a l _ { \mathcal { W } } ( w ) : = \left( \forall n \in N i g h t s . \mathsf { R o o m H e l d } ( w , n ) \right) \land \forall \mathsf { i s a G r a n t e d } ( w ) ,\tag{E.5}
$$

where the second conjunct means that the traveller’s visa has been granted. Under its hypotheses, Corollary 4 ensures that these conditions hold when settlement begins. A goal expressed through recorded confirmations supports this conclusion only if correspondence connects those records to reservations still valid at that instant.

## F What goes wrong without isolation

The examples of Appendix E show that relating the contract’s state to the world requires conditions on the routines and on the services they use. In the payment example, the argument relies on every issued instruction being backed by a ledger entry and on the provider respecting its spending limit. To apply that argument to a system used by an agent, we must also consider the access that the agent has to the account and the ledger. If it can issue a payment directly or erase an entry, the fact that the Airlock’s payment routine follows the prescribed procedure is no longer suficient.

We are interested in systems where this access can be restricted by design. For instance, keeping the payment credentials and ledger under the Airlock’s control can ensure that instructions are issued only by its routines. The tool model $\mathsf { D } _ { \mathsf { T } }$ can then describe the state changes caused by these operations without also including arbitrary modifications of the ledger by the agent. The obligation for a proposed payment checks whether the modeled changes respect the budget, while the provider’s spending limits connect the recorded amounts to actual charges. Sec. 6.1 develops capability modules, which combine access to a resource with guarantees about the operations performed on it. This appendix examines why protecting this access matters, by following what an action outside the proof gate does to the assumptions of Theorem 3, as illustrated in Figure 5.

Ungated interactions. Consider an agent that can invoke a routine in two ways. It can submit a request to the Airlock, which checks the corresponding obligation before executing the call, or invoke the routine directly without providing a proof. We call the second form of access ungated. To see how it can afect the guarantees, suppose that an invocation authorized by the Airlock starts from a compliant active state � with $\rho ( w , s )$ . While it runs, an ungated action makes the world unsafe, so that $\neg \mathsf { S a f e } ( w ^ { \prime } )$ . The running invocation may be control, as in Figure $^ { 5 , }$ or a requested tool. In either case, $\widehat { E } _ { \mathrm { T } }$ includes this change, since it describes everything that can happen to the world and state during the invocation.

If the state remains unchanged, it still satisfies ${ \mathsf { C } } ( s )$ . Under $\begin{array} { r } { \boxed { \mathsf { A } 6 } , } \end{array}$ it can therefore no longer describe the unsafe world:

$$
\mathsf C ( s ) \wedge \neg \mathsf S \mathsf { a f e } ( w ^ { \prime } ) \quad \Longrightarrow \quad \neg \rho ( w ^ { \prime } , s ) .\tag{F.1}
$$

This violates fidelity, A5 , because correspondence held at the start of the enabled invocation and has been lost while it runs. Detecting the change afterwards cannot restore a guarantee that was required to hold at every instant.

If instead the state changes to $s ^ { \prime }$ and $\rho ( w ^ { \prime } , s ^ { \prime } )$ holds, adequacy requires $\neg { \mathsf { C } } ( s ^ { \prime } )$ . The question is then whether the tool model $\mathsf { D } _ { \mathsf { T } }$ includes this state. Omitting it violates coverage, A2 , since it is a possible efect of the running invocation. Including it means that the invocation is not admissible from $s .$ For control this contradicts A3 , and for a requested tool it contradicts the admissibility established from its accepted obligation under the assumptions of Theorem 1. Finally, if we allow $\rho ( w ^ { \prime } , s ^ { \prime } )$ to hold while keeping ${ \mathsf { C } } ( s ^ { \prime } )$ true, the unsafe world is described by a compliant state and adequacy itself fails. The three branches of Figure 5 illustrate these diferent ways in which an ungated action can invalidate the assumptions.

Restricting efects. We could try to account for ungated access by enlarging $\mathsf { D } _ { \mathsf { T } }$ to cover all the state changes it makes possible. Suppose, however, that these changes include some $s ^ { \prime }$ with $E \tau ( i , s , s ^ { \prime } )$ and $\neg { \mathsf { C } } ( s ^ { \prime } )$ . For any model that includes this outcome, $\mathcal { A } _ { \sf T } ( i , s )$ is false, because admissibility requires every state described by the model to satisfy C. If this happens for every requested call throughout execution, the proof gate rejects all of them. Provided control remains admissible as required by A3 and the other hypotheses of Theorem 3 hold, the system still remains safe. It can continue running control, but cannot execute any requested action to carry out the task. Thus covering all possible behaviour can preserve the safety guarantee while making the system unusable for its intended purpose.

A separate dificulty arises when the ungated action changes the world without changing the state. Such an action appears in $\widehat { E } _ { \top }$ , even if it introduces no additional state into $E _ { \mathsf { T } }$ As (F.1) shows, a compliant state cannot remain a faithful description of an unsafe world under adequacy. Enlarging $\mathsf { D } _ { \mathsf { T } }$ cannot resolve this failure of fidelity. This is why we consider restrictions on the agent’s actual access, alongside conditions on the models used by the checker. Sec. 5.2 states the corresponding restriction as A7 .

## G A formal account of isolation

Lemma 2 allows coverage and fidelity established for reference executions to apply when an agent is present. The reason is that, during an invocation, the agent cannot change the protected system or introduce new possibilities for its evolution. This appendix gives a stepwise formulation of that argument. It uses the method of proving non-interference through conditions on individual steps (Goguen and Meseguer, 1984; Rushby, 1992).

We fix ${ \mathsf { T } } \in { \mathcal { T } } \cup \{ { \mathsf { c t r } } { \mathsf { l } } \}$ , an input � and an invocation of its routine $R = \tt e x e c \tau ( i )$ . The routine runs alongside the environment, the host and the agent. The environment includes external services and the passage of time. The host includes the Airlock’s loop and checker. As in Section 5.2, reference executions suppress the agent’s steps while leaving the other processes present.

```latex
Definition G.1 Configurations and invocation prefixes
A configuration is a tuple ${ \boldsymbol { c } } = ( \boldsymbol { w } , \boldsymbol { s } , \boldsymbol { r } , \boldsymbol { u } , \boldsymbol { a } )$ , where $w \in \mathcal { W } , s \in \Sigma ,$ and $r , u$ and � are the
local states of the routine, host and agent, respectively. The routine state � contains its
fixed input and records whether it has returned. For each actor $b \in \{ R ,$ env, host, ag},
let $ _ { b }$ be its step relation on configurations. An invocation prefix is a finite sequence
$c _ { 0 }  _ { b _ { 1 } } c _ { 1 } \cdot \cdot \cdot  _ { b _ { m } } c _ { m }$ starting with the routine in its initial state on $i ,$ in which the routine
has not returned at any $c _ { j }$ with $j < m ,$ The protected part of a configuration is
�<sub>�</sub> (�, �, �, �, �) := (�, �, �).
A reference prefix is an invocation prefix containing no step of $\mathsf { a g . }$ . Write $c $ � for a
finite sequence, possibly empty, of steps of $R ,$ env and host from � to � within the same
invocation, with no step after the routine has returned.
```

A prefix may end before the routine returns, or at its return. This is necessary because the efect relations describe every state reached during execution. When the step model represents the executions of Section 5.1, each world–state pair reached at any instant must be represented by a finite invocation prefix. In particular, a step that advances time must allow prefixes ending at intermediate instants if the world or state changes during that step. Sampling only the entry and return of a routine would not describe the efects used in the main results.

For an agent $\alpha ,$ whose processes determine $ _ { \mathsf { a g } }$ , let $\widehat { E } _ { \top } ^ { \alpha } ( i , w , s , w ^ { \prime } , s ^ { \prime } )$ hold when an invocation prefix starting with world $w$ and state � ends with world $w ^ { \prime }$ and state $s ^ { \prime }$ . Define $\widehat { E } _ { \intercal } ^ { \bullet }$ in the same way using reference prefixes. The state efects are their projections, as in $( 1 4 )$ . The initial configurations and the step relations of the other actors are the same in both cases.

Conditions on steps. The following conditions give a suficient stepwise formulation of isolation for this model. They are required for each tool and input, for configurations of the same invocation and steps taken before the routine returns.

(NI1) Preservation by the agent. If $c \to _ { \mathsf { a g } } c ^ { \prime } .$ , then $\kappa _ { R } ( c ^ { \prime } ) = \kappa _ { R } ( c )$

(NI2) Independence of the protected evolution. If $c  _ { b } c ^ { \prime }$ with $b \neq \mathsf { a g }$ and $\kappa _ { R } ( d ) = \kappa _ { R } ( c )$ , then there is a configuration $d ^ { \prime }$ such that $d  d ^ { \prime }$ and $\kappa _ { R } ( d ^ { \prime } ) = \kappa _ { R } ( c ^ { \prime } )$

(NI1) expresses the restriction on the agent’s access in A7 : its computation and submission of requests leave the running routine, world and state unchanged. (NI2) expresses the independence needed to remove that computation. When two configurations agree on the protected quantities, a change produced by the other actors from one must also be possible from the other without an agent step. The matching sequence may be empty when the original step leaves the protected quantities unchanged. Thus, for example, a checker’s private progress need not be reproduced step for step.

These conditions must be established for the interfaces and execution environment that enforce isolation. They do not require knowing how the agent chooses its requests or proofs. Host and service actions remain in reference executions, and their efects must still be covered by the tool models. Execution integrity, A4 , continues to govern the scheduling of invocations and the connection between requests, views and calls.

## Lemma G.1 Isolation, stepwise formulation

For any agent � satisfying (NI1) and (NI2), every invocation prefix has a reference prefix starting from the same configuration and ending in a configuration with the same protected part. Consequently, for all inputs, worlds and states,

$$
\widehat { E } _ { \mathbb { T } } ^ { \alpha } ( i , w , s , w ^ { \prime } , s ^ { \prime } ) \implies \widehat { E } _ { \mathbb { T } } ^ { \bullet } ( i , w , s , w ^ { \prime } , s ^ { \prime } ) , \qquad E _ { \mathbb { T } } ^ { \alpha } ( i , s , s ^ { \prime } ) \implies E _ { \mathbb { T } } ^ { \bullet } ( i , s , s ^ { \prime } ) .
$$

Proof. Consider an invocation prefix $c _ { 0 }  _ { b _ { 1 } } c _ { 1 } \cdot \cdot \cdot  _ { b _ { m } } c _ { m }$ . We construct a reference prefix starting at $c _ { 0 }$ whose endpoint $d _ { j }$ satisfies $\kappa _ { R } ( d _ { j } ) = \kappa _ { R } ( c _ { j } )$ , by induction on �.

For $j = 0 ,$ , the empty prefix sufices. Suppose the prefix has been constructed up to $j < m$ . If $b _ { j + 1 } = \mathsf { a g } _ { \mathrm { : } }$ , (NI1) gives $\kappa _ { R } ( c _ { j + 1 } ) \ = \ \kappa _ { R } ( c _ { j } )$ , so the same reference prefix serves, with $d _ { j + 1 } = d _ { j }$ Otherwise, (NI2) gives a sequence $d _ { j }  d _ { j + 1 }$ with $\kappa _ { R } \big ( d _ { j + 1 } \big ) = \kappa _ { R } \big ( c _ { j + 1 } \big )$ , which we append. Since $j < m _ { : }$ , the routine has not returned at $c _ { j }$ or at its matching configuration $d _ { j }$ . The appended sequence takes no step after return, so the concatenation is a reference prefix.

$\operatorname { A t } j = m$ , the constructed prefix starts at the same invocation and has the same final world and state. It therefore witnesses the implication for ${ \widehat { E } } .$ Projection on states gives the implication for �. ■

Thus coverage and both parts of fidelity for reference executions also hold for actual executions, as required by Lemma 2.