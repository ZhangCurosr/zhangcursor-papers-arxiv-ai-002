# Making Duplicate Reimbursement Unrepresentable: A Verified Ethereum E-Invoice System for Humans and AI Agents

Jia Cai

College of Engineering and Computing

George Mason UNiversity

Fairfax, VA, USA

jcai8@gmu.edu

Abstract—Electronic invoices are replacing paper invoices worldwide, but today’s centralized e-invoice architectures leave three problems unsolved on the consumption side: an e-invoice can be printed and submitted for reimbursement repeatedly, its authenticity is hard for recipients to verify, and invoice data is siloed at a central authority that becomes both a performance bottleneck and a single point of failure. This paper presents the design, formal analysis, and implementation of a complete blockchain-based electronic invoice system on Ethereum. We give a formal model of the invoice lifecycle as a guarded labeled transition system and prove, under standard cryptographic and consensus assumptions, that the system guarantees (i) reimbursement uniqueness—an invoice can be reimbursed at most once, even across mutually distrusting organizations; (ii) face integrity— any invoice that passes verification equals the recorded one unless keccak256 second-preimage resistance is broken; and (iii) authorization soundness for every lifecycle operation. The core state-machine invariants are additionally machine-checked with the Solidity SMTChecker, which proves them inductively over all reachable transaction sequences. The design models each invoice as a non-fungible, non-tradable token whose ownership and state change only through five lifecycle subsystems, and a lockbased reimbursement protocol makes duplicate reimbursement unrepresentable rather than merely detectable. We implement the design as a Solidity 0.8 smart contract with a four-role web application and evaluate it on a private Ethereum network: issuing an invoice costs 646,773 gas, the complete reimbursement protocol costs under 135,000 gas, every core operation is O(1) in the number of invoices, and a single development node sustains 137 invoice issuances per second. Finally, we show that the verified contract doubles as a safety envelope for autonomous AI agents: an LLM-based reimbursement agent operating a registered account is exactly the adversary of our threat model, so agent safety follows as a corollary of the proved theorems, and in an end-to-end case study every unsafe agent action — duplicate, over-limit, or forged-receipt claims, even with the agent’s own policy layer bypassed — is rejected, ultimately by the contract itself. The full implementation, agent runtime, test suite, and benchmarks are open source. The results indicate that lifecycle-complete, formally-grounded invoice management on a blockchain is practical and directly eliminates the duplicatereimbursement and verification pain points of centralized designs.

Index Terms—blockchain, Ethereum, smart contract, electronic invoice, non-fungible token, formal model, autonomous agent, tax administration

## I. INTRODUCTION

Invoices are the primary written evidence of commercial transactions. They anchor bookkeeping and expense reimbursement inside firms, value-added tax (VAT) collection by tax authorities, audit by regulators, and evidence in litigation. Their digitization is accelerating globally: China has operated a nationwide VAT e-invoice system since 2015 [1], and the European Union’s “VAT in the Digital Age” package, adopted in March 2025, mandates structured electronic invoicing for intra-EU B2B transactions by 2030, projecting fraud reductions of up to C11 billion per year [2], [3].

Existing e-invoice systems are centralized: a tax authority (or its service provider) issues, records, and validates all invoices. This architecture digitizes the issuing side effectively but neglects the consumption side, where three pain points persist.

Duplicate reimbursement. An electronic invoice is a file that can be printed or forwarded any number of times. An employee can submit the same invoice for reimbursement at two employers, or twice at the same employer through different channels. Finance departments today defend against this with manually maintained registers of already-reimbursed invoice numbers—a process that is error-prone, unauditable, and does not work across organizations at all.

Costly verification. Paper invoices carried physical antiforgery features; a printed e-invoice carries none. Recipients must query the authority’s central platform to verify authenticity, and in practice most do not, so forged and altered einvoices circulate. Some firms respond by refusing e-invoices above certain amounts, undermining adoption.

Central bottleneck and data silo. All invoice data converges on the authority’s platform, which must be provisioned for peak nationwide load, becomes a single point of failure, and gives downstream users (buyers, auditors, banks, courts) no direct, tamper-evident access to invoice state.

These pain points are precisely the trust and state-sharing problems that blockchains address [4], [5]. A blockchain provides a replicated, append-only ledger whose state transitions are validated by consensus rather than by a single operator; a smart contract platform such as Ethereum [6], [7] additionally lets the invoice lifecycle itself be encoded as executable rules that no participant can bypass. If every invoice is a unique on-chain asset whose reimbursement is a one-way state transition, duplicate reimbursement is not merely detectable— it is unrepresentable.

This paper designs, formally analyzes, implements, and evaluates such a system. Concretely, we make six contributions:

1) Requirements and architecture. We analyze the business requirements of electronic invoicing from the perspectives of the tax authority, the invoice issuer (seller), the recipient (buyer), and third-party verifiers, and derive a five-subsystem architecture covering the complete invoice lifecycle (Section V).

2) Formal model. We formalize the system as a guarded labeled transition system over an explicit global state, with an invoice lifecycle automaton and a complete table of transition rules (Section IV).

3) Security proofs, machine-checked. Under an explicit threat model, we prove reimbursement uniqueness, face integrity, red-flush value conservation, and authorization soundness, and mechanically verify the core statemachine invariants with the Solidity SMTChecker (Section VII).

4) Complete implementation. We realize the design as a single auditable Solidity 0.8.24 contract that models invoices as non-fungible, non-tradable tokens, together with a four-role web application, released as open source (Sections VI–VIII).

5) Evaluation. We evaluate functional correctness (23 unit tests spanning all subsystems), per-operation gas cost, asymptotic complexity, latency, and throughput on a private Ethereum network (Section IX).

6) Agentic case study. We build an autonomous reimbursement agent (hybrid deterministic/LLM reasoning) and a ledger-audit agent on top of the contract, show that agent safety is a corollary of the proved theorems — the agent is a registered account, i.e., the modeled adversary — and demonstrate end-to-end that unsafe agent actions are rejected even when the agent’s own policy layer is bypassed (Section X).

The rest of the paper is organized as follows. Section II surveys related work and positions our contribution. Section III gives preliminaries and the threat model. Section IV presents the formal model. Sections V–VI give the architecture and contract design; Section VII proves the security properties; Sections VIII–IX cover implementation and evaluation; Section X presents the autonomous-agent layer and its case study; Sections XI–XII discuss limitations and conclude.

## II. RELATED WORK

Blockchain platforms. Bitcoin introduced the replicated append-only transaction ledger secured by proof of work [8]. Ethereum generalized the model with a Turing-complete virtual machine and persistent contract state [6], [7], enabling applications beyond currency transfer [5]. Consortium deployments typically replace proof of work with committee-based consensus such as PBFT [9], which suits closed ecosystems like taxation, where participants are identified and permissioned. Our contract layer is consensus-agnostic and runs unmodified on either configuration.

Blockchain for tax and invoicing. Hyvarinen et al. pro-¨ totyped a blockchain service that eliminates dividend-tax refund fraud in Denmark by making refund state explicit and shared [10]—the same “make fraud unrepresentable” principle we apply to reimbursement. Fatz et al. proposed decentralized validation of VAT processes to achieve tax compliance by design [11]. Zhang and Liu analyzed the data-sharing characteristics of e-invoices and sketched a blockchain einvoice design, without a lifecycle implementation [12]. On the industrial side, the Shenzhen Municipal Taxation Bureau and Tencent launched China’s first production blockchain einvoice in August 2018 [13], issuing about six million invoices in the first year [14]; the system is proprietary, and its contractlevel design is not public.

Smart-contract e-invoice and VAT systems. Closest to our work, Nguyen et al. digitize invoices with an Ethereum smart contract combined with a decentralized storage network, authenticating a transaction and then computing and approving its VAT payment [15]. More recently, Farchan implemented e-invoice (e-Faktur) issuance and validation for Indonesia’s VAT system on a permissioned Hyperledger Fabric ledger, evaluating resilience against simulated attacks [16]. Both target the issuing and reporting side—authenticating an invoice and settling its VAT—and neither addresses the consumption-side lifecycle: neither models invoices as ownable tokens, implements red-flush credit-note correction, nor prevents duplicate reimbursement. A parallel, largely industry-driven line of work tokenizes invoices as tradable ERC-721 assets [17] for invoice financing, where the analogous hazard is double-financing the same receivable; there tradability is the whole point, the opposite of our non-tradable design for legal tax documents.

Positioning. Table I summarizes the comparison. Relative to prior academic work, which is largely conceptual, targets a single fraud scenario, or covers only invoice issuance and VAT settlement, this paper contributes a lifecycle-complete, formally-analyzed, open-source design— from blank-invoice distribution through red-flush correction to lock-based reimbursement—with proved security properties and reproducible cost measurements. Relative to proprietary industrial systems, it provides a published, auditable contract design; we follow established Solidity security patterns [18], [19].

## III. PRELIMINARIES AND THREAT MODEL

## A. Blockchain and Smart Contracts

Ethereum maintains a replicated state machine: externally owned accounts hold balances and issue signed transactions, and contract accounts hold code and storage executed by the Ethereum Virtual Machine (EVM) [7]. A smart contract is a program whose functions are invoked by transactions; every state-changing invocation is ordered by consensus, recorded immutably, and metered in gas, which measures computational and storage cost independently of any token price. Contracts emit events—indexed log entries that clients query to reconstruct a complete, tamper-evident history of an asset.

TABLE I  
COMPARISON WITH REPRESENTATIVE BLOCKCHAIN E-INVOICE / TAX SYSTEMS. ✓ = SUPPORTED, ▲ = PARTIAL OR PROPRIETARY, ✗ = NOT ADDRESSED.
<table><tr><td>System</td><td>Platform</td><td>Invoice model</td><td>Lifecycle coverage</td><td>Digest verification</td><td>Red-flush</td><td>Duplicate-reimburse prevention</td><td>Formal proofs open source</td></tr><tr><td>Zhang &amp; Liu &#x27;17 [12]</td><td>Conceptual</td><td>data record</td><td>issuance (sketch)</td><td></td><td>x</td><td>x</td><td>X / X</td></tr><tr><td>Hyvärinen et al. &#x27;17 [10]</td><td>Ethereum</td><td>refund claim</td><td>dividend-tax refund</td><td></td><td>X</td><td> $\checkmark \mathrm { ( r e f u n d ) }$ </td><td>X / X</td></tr><tr><td>Shenzhen/Tencent &#x27;18 [13]</td><td>Consortium</td><td>data record</td><td>issue → reimburse</td><td></td><td></td><td> $\pm \ ( \mathrm { c l o s e d } )$ </td><td>X / X</td></tr><tr><td>Fatz et al. &#x27;19 [11]</td><td>Ethereum</td><td>process instance</td><td>VAT process validation</td><td></td><td>X</td><td>x</td><td>X / X</td></tr><tr><td>Nguyen et al. &#x27;19 [15]</td><td>Ethereum + DSN</td><td>data record</td><td>issue + VAT payment</td><td></td><td>x</td><td>x</td><td>X/x</td></tr><tr><td>Farchan &#x27;24 [16]</td><td>Hyperledger Fabric</td><td>data record</td><td>issue + reporting</td><td></td><td></td><td>x</td><td>X / X</td></tr><tr><td>This work</td><td>Ethereum / EVM</td><td>non-tradable NFT</td><td>full five-subsystem</td><td></td><td></td><td>√(cross-org)</td><td>√1√</td></tr></table>

Two properties are load-bearing for invoicing. First, storage writes are permanent and globally replicated, so an on-chain invoice cannot be silently altered or deleted; corrections must themselves be recorded transactions. Second, contract code enforces its own invariants: if the contract exposes no function that reimburses an invoice twice, no participant—including the operator—can do so.

## B. Cryptographic Primitives

We use the keccak256 hash function ${ \cal H } ~ : ~ \{ 0 , 1 \} ^ { * } ~ $ $\{ 0 , 1 \} ^ { 2 5 6 }$ modeled as second-preimage- and collisionresistant, and ECDSA signatures over secp256k1, assumed existentially unforgeable under chosen-message attack (EUF-CMA); the EVM authenticates every transaction’s sender by signature. Consensus is assumed safe (no two conflicting histories finalize) under its fault threshold—honest hash-power majority for proof of work, or fewer than $n / 3$ Byzantine nodes for PBFT-class consortium consensus [9]. Consequently, transactions apply to a single, totally ordered, globally agreed state.

## C. Threat Model

The adversary A controls the private keys of an arbitrary set of registered enterprise accounts and may submit any transactions in any order, adaptively. This explicitly includes faulty or adversarial autonomous agents: software (including LLM-based agents, Section X) operating a registered account is, from the contract’s perspective, indistinguishable from any other key holder, so every guarantee proved against A applies verbatim to arbitrary agent behavior, including behavior induced by prompt injection. A cannot forge signatures of honest accounts (EUF-CMA), cannot find collisions or second preimages of $H ,$ and cannot violate consensus safety. The tax bureau key is honest for admission (registration and blank-invoice supply), reflecting its legal role, but—unlike centralized systems—is not trusted for invoice state, which the contract and consensus enforce. A succeeds if it can (G1) cause some invoice to be reimbursed twice; (G2) present an invoice face that verifies yet differs from the recorded one; or (G3) drive a lifecycle transition on an invoice it is not authorized to act on. Section VII proves each goal infeasible. Off-chain concerns (fictitious underlying sales, key theft, storage-layer confidentiality) are out of scope for these guarantees and discussed in Section XI.

TABLE II  
NOTATION FOR THE FORMAL MODEL.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\mathcal { A }$ </td><td>address space;  $b \in { \mathcal { A } }$  the tax bureau</td></tr><tr><td> $\mathcal { N }$ </td><td>invoice-number space (nonzero)</td></tr><tr><td> $R \subseteq A$ </td><td>registered enterprise accounts</td></tr><tr><td> $O : \mathcal { N } \xrightarrow { } \mathcal { A }$ </td><td>invoice owner (undefined = nonexistent)</td></tr><tr><td> $\mathsf { s t } : \mathcal { N } \to Q$ </td><td>lifecycle status, Q from Def. 3</td></tr><tr><td> $\mathsf { c r } : \mathcal { N } \to \{ 0 , 1 \}$ </td><td>credit-note flag</td></tr><tr><td> $\mathsf { d } \mathsf { c } : \mathcal { N } \to \{ 0 , 1 \}$ </td><td>tax-declared flag</td></tr><tr><td> $F : \mathcal { N } \overset { } { \mathop {  } } \mathbb { F }$ </td><td>invoice face record (Def. 2)</td></tr><tr><td> $D : \{ 0 , 1 \} ^ { 2 5 6 }  \mathcal { N }$ </td><td>digest index</td></tr><tr><td> $L : \mathcal { N }  \mathcal { A } \times \mathsf { C i d }$ </td><td>reimbursement lock (locker, claim id)</td></tr><tr><td> ${ \mathsf { s e l } } ( n ) , { \mathsf { b u y } } ( n )$ </td><td>seller / buyer address of invoice n</td></tr><tr><td> $\tan ( n )$ </td><td>tax-inclusive total of invoice n (cents)</td></tr></table>

## IV. FORMAL MODEL

We model the contract as a deterministic transition system over a global state. Table II lists the notation.

Definition 1 (Global state). A state is a tuple $\sigma =$ $( R , O , \mathsf { s t } , \mathsf { c r } , \mathsf { d c } , F , D , L )$ . The genesis state $\sigma _ { 0 }$ has $R = \varnothing$ and all partial maps empty; ${ \mathsf { s t } } ( n ) = { \mathrm { N o N E } }$ wherever $O ( n )$ is undefined.

Definition 2 (Invoice face). A face $F ( n ) \in \mathbb { F }$ is the record ⟨sel, buy (each a snapshot of taxpayer id, name, bank, address), $p , \ r , \ t ,$ tot, $\kappa ,$ items, ${ \mathsf { c r } } , \ \tau \rangle$ , where p is the pretax amount in cents, r the rate in basis points, $t = \lfloor p \cdot r / 1 0 ^ { 4 } \rfloor$ the tax, tot $= p + t ,$ κ the category code, and τ the issuance time. Its digest is $H ( F ( n ) )$ over the canonical encoding of ⟨n, sel.id, buy.id, $p , r , \kappa , i t e m s , { \mathsf { c r } } , \tau \rangle$

Definition 3 (Lifecycle transition system). The invoice lifecycle is the labeled transition system $\begin{array} { r l r } { \mathcal { L } } & { { } = } & { \left( Q , \Sigma , - , \mathrm { N o N E } \right) } \end{array}$ with states $\begin{array} { r l } { Q } & { { } = } \end{array}$ {NONE, BLANK, ISSUED, LOCKED, REIMBURSED, REVERSED}, labels Σ the contract operations of Table III, and → the

![](images/21a86de03651d8955e69e50b577fa0f124d4dd9b1b49517a270bbd25eb370f67.jpg)  
Fig. 1. Invoice lifecycle automaton L (Def. 3). Double-bordered red states are terminal; the gold node is a frozen credit note. Tax declaration is an orthogonal one-way flag dc and is omitted for clarity.

per-invoice projection of those rules (Fig. 1). REIMBURSED and REVERSED are terminal (no outgoing edges).

Each operation is a guarded command op(c, ⃗x) : $g u a r d ( \sigma , c , \vec { x } ) \Rightarrow \sigma ^ { \prime } .$ where c is the transaction sender; if the guard fails the transaction reverts and σ is unchanged. Table III gives the guards and effects; these are exactly the require clauses and state writes of the contract. An execution is a finite sequence $\sigma _ { 0 } \xrightarrow { \mathsf { o p } _ { 1 } } \sigma _ { 1 } \cdots ;$ by consensus safety it is a single total order.

## V. REQUIREMENTS AND SYSTEM ARCHITECTURE

## A. Actors and Functional Requirements

Four actors participate in the invoice lifecycle:

• Tax bureau: the sole authority that admits enterprises and creates blank invoices; monitors all invoice state.

• Seller (issuer): applies for blank invoices, issues invoices for real transactions, corrects erroneous invoices, declares output tax.

• Buyer (recipient): receives invoices, verifies authenticity, reimburses expenses against invoices.

• Third-party verifier: auditors, courts, banks, and other parties who must verify invoices without trusting seller or buyer.

From the pain points in Section III and current invoice regulations we derive five functional subsystems, which together cover the complete lifecycle: (1) application and distribution—only the bureau may create invoices; enterprises apply and receive numbered ranges of blanks, and the bureau may grant fewer than requested; (2) issuance and circulation— filling a blank’s face and delivering it to the buyer is one atomic action, and the face is immutable once issued; (3) void and red-flush—erroneous invoices cannot be edited or deleted, so the original seller issues a linked negative credit invoice and the original is marked reversed; (4) query and verification— any authorized party queries invoice state by number and verifies a face’s authenticity from its content digest; (5) declaration and reimbursement—the seller declares output tax exactly once, and the buyer reimburses an invoice at most once, with an explicit failure-recovery path. Non-functional requirements include permissioned participation, auditability, throughput adequate for enterprise-scale invoicing, and no trusted intermediary on the consumption side.

![](images/0c1dcb25f72b5918323a8ab68742002d454dbc9c7f866be823cbc4a3606a84c4.jpg)  
Fig. 2. System architecture. The smart contract is the sole trust boundary; dashboards are stateless clients.

## B. Is a Blockchain Warranted?

Applying the standard applicability test—shared state, multiple writers, mutual distrust, and no perfect trusted third party—electronic invoicing qualifies on all four counts. Invoice state must be shared among bureau, seller, buyer, and verifiers; all of them write state at different lifecycle stages; sellers and buyers have adversarial incentives; and the existing trusted third party, the central platform, is exactly the bottleneck and silo we seek to remove. Because participants are identified enterprises under a regulator, a consortium deployment is the natural fit, with the public-chain design retained as the more adversarial baseline.

## C. Architecture

Fig. 2 shows the architecture. All lifecycle rules reside in a single smart contract (the trust boundary); role dashboards are thin clients that submit signed transactions and reconstruct history from events. No application server holds authoritative state.

## VI. SMART CONTRACT DESIGN

## A. Invoices as Non-Tradable, Non-Fungible Tokens

Each invoice is unique, indivisible, and identified by its number, which naturally suggests the non-fungible token model of ERC-721 [17]: the contract maintains invoiceOwner (number → holder), per-holder invoice sets, and mint/transfer primitives. We deliberately deviate from ERC-721 in one crucial respect: there is no public transfer function. Invoices are legal documents, not tradable assets; ownership changes only as a side effect of lifecycle operations (distribution moves a blank from bureau to seller; issuance moves the invoice from seller to buyer). This closes an entire class of misuse (invoice resale, gray-market circulation) at the type-system level.

TABLE III  
GUARDED TRANSITION RULES (PER INVOICE n). c IS THE CALLER. ALL GUARDS ARE CONJUNCTIVE; A FAILED GUARD REVERTS WITH NO EFFECT.
<table><tr><td>Operation</td><td>Guard (precondition)</td><td>Effect (state update)</td></tr><tr><td> $\mathsf { r e g i s t e r } ( c , a , i d )$ </td><td> $c = b \land a \notin R$ </td><td>R += a</td></tr><tr><td>mint(c, n, a)</td><td> $c = b \land a \in R \land O ( n ) { = } \bot$ </td><td> $O ( n ) { : = } a { \mathrm { ; } } \ { \mathsf { s t } } ( n ) { : = } { \mathbf { B } } { \mathrm { L A N K } }$ </td></tr><tr><td> $\mathsf { i s s u e } ( c , n , v , \phi )$ </td><td> $c \in R \land O ( n ) { = } c \land \mathsf { s t } ( n ) { = } \mathbf { B } \mathbf { L A N K } \land v \in R \land v \neq c \land v a l i d ( \phi )$ </td><td> $F ( n ) { : = } \phi ; \ { \mathsf { s t } } ( n ) { : = } { \mathsf { I s s U E D } } ; \ O ( n ) { : = } v ; \ D ( H ( \phi ) ) { : = } n$   $F ( m ) { : = } F ( n ) , \ \mathsf { c r } ( m ) { : = } 1 ; \ \mathsf { s t } ( n ) { : = } \mathrm { R E v E R S E D } ;$ </td></tr><tr><td>redFlush(c, n, m)</td><td> $c \in R \land \mathsf { s t } ( n ) = \operatorname { I s s U E D } \land \mathsf { c r } ( n ) = 0 \land \mathsf { s e l } ( n ) = c \land O ( m ) = c \land \mathsf { s t } ( m ) = \mathsf { B L A N K }$ </td><td> $\mathsf { s t } ( m ) { \mathrel { \mathop : } } = \mathrm { I s s U E D } ; O ( m ) { \mathrel { \mathop : } } = \mathsf { b u y } ( n ) ; D ( H ( F ( m ) ) ) { \mathrel { \mathop : } } = m$ </td></tr><tr><td>declare(c, n)</td><td> $c = { \mathsf { s e l } } ( n ) \wedge { \mathsf { s t } } ( n ) \not \in \{ { \mathrm { N O N E } } , { \mathrm { B L A N K } } \} \wedge { \mathsf { c r } } ( n ) { \mathsf { = 0 } } \wedge { \mathsf { d c } } ( n ) { \mathsf { = 0 } }$ </td><td>dc(n):=1</td></tr><tr><td>lock(c, n, cid)</td><td> $\mathsf { s t } ( n ) { = } \mathrm { I s s U E D } \wedge \mathsf { c r } ( n ) { = } 0 \wedge O ( n ) { = } c [ \wedge \mathsf { d } \mathsf { c } ( n ) { = } 1 ]$ </td><td> $\mathsf { s t } ( n ) { : = } \mathrm { L o c K E D } ; L ( n ) { : = } ( c , c i d )$ </td></tr><tr><td>reimburse(c, n)</td><td> $\mathsf { s t } ( n ) { = } \mathrm { L o c k E D } \wedge L ( n ) . l o c k e r { = } c$ </td><td> $\mathsf { s t } ( n ) { : = } \mathsf { R E I M B U R S E D }$ </td></tr><tr><td>unlock(c, n)</td><td> $\mathsf { s t } ( n ) { = } \mathrm { L o c k E D } \wedge L ( n ) . l o c k e r { = } c$ </td><td> $\mathsf { s t } ( n ) { : = } \mathrm { I s s U E D } ; L ( n ) { : = } \bot$ </td></tr></table>

## B. Data Model

The contract stores, per invoice, the fields of Def. 2: the two parties (address plus a snapshot of taxpayer id, legal name, bank information, and registered address, copied from the enterprise registry at issuance time so the face stays immutable even if the registry later changes), the amounts, the category code and item description, the lifecycle status, the credit flag, a bidirectional linkage field for red-flush, the issuance timestamp, and the face digest. An enterprise registry, writable only by the bureau, binds each participating address to its taxpayer identity; every lifecycle operation checks registration, giving permissioned semantics even on a public chain. Monetary values are unsigned integers (cents); credit invoices carry positive magnitudes plus the isCredit flag rather than signed values, avoiding sign-handling errors. Tax is computed on chain as ⌊preTax · rateBps/10<sup>4</sup>⌋, so an arithmetically inconsistent face cannot exist.

## C. Subsystem 1: Application and Distribution

An enterprise calls applyForInvoices(count); the bureau either rejects or calls approveApplication(id, startId, granted) with granted ≤ count, minting the range [startId, startId+granted) of blanks directly to the applicant. Uniqueness is enforced at mint time. Because distribution is a ledger transition rather than a portal download, the bureau needs no high-availability distribution platform, and every outstanding blank is publicly attributable to its holder.

## D. Subsystem 2: Issuance and Circulation

issueInvoice (Listing 1) checks that the caller holds the blank, that both parties are registered and distinct, and that amounts are valid; it then snapshots both parties, computes tax, stores the face, records the face digest, and transfers ownership to the buyer—one atomic transaction. Delivery and recording are inseparable: the buyer’s very possession of the invoice implies a validated on-chain record.

## E. Subsystem 3: Void and Red-Flush

Ledger immutability forbids editing an issued invoice, so correction is itself a recorded transaction. redFlush(originalId, blankId) (Fig. 3) may be called only by the original seller, only on an invoice still

```solidity
function issueInvoice(uint256 id, address buyer,
uint256 preTax, uint256 rateBps,
uint256 category, string calldata items)
external onlyRegistered {
require(invoiceOwner[id] == msg.sender);
require(invoices[id].status == Status.Blank);
require(enterprises[buyer].registered
&& buyer != msg.sender);
require(preTax > 0 && rateBps <= 10000);
_fillFace(invoices[id], msg.sender, buyer,
preTax, rateBps, category, items, false);
_transfer(msg.sender, buyer, id);
emit InvoiceIssued(id, msg.sender, buyer,
invoices[id].totalAmount,
invoices[id].contentHash);
}
```  
Listing 1. Issuance (abridged).

Issued, and only using a blank the seller holds; it copies the face onto the credit note with isCredit set, links both invoices bidirectionally, marks the original Reversed, and delivers the credit note to the original buyer. Both the error and its correction remain permanently visible, eliminating the “void and reprint” fraud pattern of centralized systems.

## F. Subsystem 4: Query and Verification

At issuance the contract computes $h = H ( \cdot )$ over the canonical face encoding (Def. 2) and stores h 7→ id in a digest index. A verifier holding a purported face recomputes h<sup>′</sup> locally and calls the read-only verifyByHash(h<sup>′</sup>): a match proves the face is exactly what the issuing transaction recorded; any alteration of any field yields a digest absent from the index (Theorem 3). Verification requires no interaction with—or trust in—seller, buyer, or any central platform. Standard queries and the event log complete the audit surface: the full history of an invoice is reconstructible from indexed events alone, which our verifier dashboard demonstrates.

## G. Subsystem 5: Declaration and Lock-Based Reimbursement

The seller’s declareTax marks an invoice’s output tax declared, exactly once, and only by the seller; a bureau policy switch can require declaration before reimbursement, encoding at contract level a rule that today exists only on paper. Reimbursement is the pain point that motivates the system, implemented as a two-phase protocol (Listing 2,

![](images/880a278c9635ca18ebb373d911f08e06f5846832feee6374891a4d3d01eb69fc.jpg)  
Fig. 3. Red-flush control flow. Any failed guard reverts, leaving both invoices unchanged.

Fig. 4) following the principle of making the invalid unrepresentable: lockForReimbursement moves an Issued, non-credit invoice held by the caller to Locked, recording locker and claim id; reimburse moves a Locked invoice to the terminal Reimbursed, callable only by the locker; unlockReimbursement lets only the locker release a failed claim back to Issued. The lock phase exists because reimbursement is a workflow, not an instant: between claim submission and approval the invoice must be unavailable to any competing claim yet recoverable if the claim fails. Binding the lock to a claim-document id also gives auditors a direct on-chain join between invoices and expense claims.

## VII. SECURITY ANALYSIS

We prove that the goals of the threat model (Section III-C) are infeasible. Throughout, “for invoice n” quantifies over any single number, and executions are the totally ordered sequences of Def. 3. We first record an invariant.

![](images/953c1edfafe741337ca791f293f8c122a3cf817024053f957cfea1e274d6434a.jpg)  
Listing 2. Lock-based reimbursement (abridged).

![](images/ba05725ffc84d0c4a174bd1d7af305e7fd8952a54fcea1136fe275a55c1b3bd5.jpg)  
Fig. 4. Reimbursement sequence. The duplicate lock is rejected by the contract; any third party can independently verify and audit.

Lemma 1 (Status monotonicity). In any execution, once st(n) = REIMBURSED or st(n) = REVERSED, no subsequent transition changes st(n).

Proof. By inspection of Table III, the only operations that write st(n) are mint, issue, redFlush, lock, reimburse, unlock, and their guards require st(n) ∈ {NONE}, {BLANK}, {ISSUED} (for the original) or {BLANK} (for the credit slot m), {ISSUED}, {LOCKED}, and {LOCKED} respectively. None is satisfiable when st(n) ∈ {REIMBURSED, REVERSED}. Hence these states are sinks. □

Theorem 1 (Reimbursement uniqueness). In any execution starting from σ<sub>0</sub>, for every invoice n the operation reimburse(·, n) occurs at most once.

Proof. reimburse(·, n) has guard st(n) = LOCKED and effect st(n):=REIMBURSED. Suppose it occurs at step i. By Lemma 1, st(n) = REIMBURSED at every step j > i. For a second occurrence at some j > i its guard would require $\mathsf { s t } ( n ) = \mathrm { L o c K E D } \neq$ REIMBURSED, a contradiction. Hence at most one occurrence. □

Corollary 1 (Cross-organization non-duplication). No two reimbursements of the same invoice can succeed regardless of how many distinct organizations or accounts attempt them, and regardless of concurrency.

Proof. By consensus safety the concurrent attempts are serialized into one execution (Section III); Theorem 1 applies to that execution. The guarantee is a property of the single shared slot st(n), not of any per-organization bookkeeping, so it is independent of the callers’ identities or affiliations. In a race, the first lock sets LOCKED and every later lock sees st(n) ̸= ISSUED and reverts. □

Theorem 2 (Reimbursement authorization). Ifreimburse(c, n) succeeds, then c is the account that most recently locked n and held n at that time, and n was not unlocked in between.

Proof. The guard requires $\begin{array} { r l r } { { \mathsf { s t } } ( n ) } & { { } = } & { { \mathsf { L o c K E D } } } \end{array}$ and $L ( n ) . l o c k e r = c . \ L ( n )$ is written only by lock, to (caller , cid) under the guard $O ( n ) = c a l l e r$ , and cleared only by unlock (which also sets ISSUED). Thus whenever $\mathsf { s t } ( n ) = \mathrm { L o c } \mathrm { K E D } ,$ $L ( n )$ .locker equals the caller of the most recent lock, who held n then, and no intervening unlock occurred (else $\mathsf { s t } ( n ) = \mathrm { I s s U E D } )$ . By EUF-CMA the sender field c cannot be spoofed. □

Theorem 3 (Face integrity). Let ${ \mathsf { v e r i f y } } ( \phi ) \triangleq H ( \phi ) \in$ dom(D). If a PPT adversary outputs a face ϕ with verif $\prime ( \phi ) =$ true and $\phi \neq F ( D ( H ( \phi ) ) )$ , then it has computed a second preimage of H. Hence, under second-preimage resistance, any face that verifies equals the recorded face except with negligible probability.

Proof. D is written only by issue and redFlush, each as $D ( H ( \phi ^ { \star } ) ) { : = } n$ for the face $\phi ^ { \star } = F ( n )$ actually stored. So every $h \in \mathrm { d o m } ( D )$ satisfies $h = H ( F ( D ( h ) ) )$ ). If verify(ϕ) holds then $H ( \phi ) \ : = \ : H ( F ( n ) )$ for $n = D ( H ( \phi ) )$ . If additionally $\phi \neq F ( n )$ , then $\left( \phi , F ( n ) \right)$ is a colliding pair with $H ( \phi ) = H ( F ( n ) ) ;$ ; given the recorded target $F ( n )$ this is a second preimage. □

Theorem 4 (Red-flush value conservation). After redF $\mathsf { I u s h } ( c , n , m )$ succeeds, the signed values $\nu ( n ) + \nu ( m ) = 0 ,$ where $\nu ( k ) = ( 1 - 2 { \mathsf { c r } } ( k ) ) \cdot { \mathsf { t o t } } ( k ) ;$ moreover n is permanently REVERSED and the credit m can never be reimbursed or further red-flushed.

Proof. The effect copies the face of n to m, so tot $( m ) =$ tot(n), and sets $\mathsf { c r } ( m ) { = } 1$ while $\mathsf { c r } ( n ) { = } 0 ;$ thus $\nu ( n ) \ =$ $+ \mathsf { t o t } ( n )$ and $\nu ( m ) = - \cot ( n )$ , summing to 0. It sets $\scriptstyle \mathtt { s t } ( n ) = \mathtt { R E V E R S E D }$ , terminal by Lemma 1. For m: lock requires $\mathsf { c r } = 0$ and fails on m; redFlush as an original requires $\mathsf { c r } = 0$ and fails, while as a credit slot it requires the slot be BLANK, but $\mathsf { s t } ( m ) = \mathrm { I s s u g e D }$ . Hence m is frozen. □

## A. Machine-Checked Verification

Theorems 1–4 are properties of the model of Section IV, whose guards and effects are transcribed directly from the contract’s require clauses and state writes. To remove the hand-proof as a single point of trust, we additionally encode the state-machine invariants as assert statements in a Solidity model that preserves the registry, ownership, and every guarded transition of Table III (abstracting only the invoice-face strings and hashing, which are orthogonal to these safety properties), and discharge them with the Solidity SMTChecker [20]. Its constrained-Horn-clause (CHC) engine proves properties inductively over all reachable transaction sequences and all invoice numbers, not merely the bounded scenarios of a test suite. The ghost counter that records how often the reimburse body executes yields the direct encoding of Theorem 1: assert(reimburseCount[n] == 1). The checker reports

## CHC: 5 verification condition(s) proved safe!

covering reimbursement uniqueness (Theorem 1), reimbursement authorization (Theorem 2), the impossibility of reimbursing a credit or reversed invoice, and red-flush value conservation and monotonicity (Theorem 4, Lemma 1), all with no counterexamples. As a non-vacuity check, weakening the reimburse guard to also accept an already-REIMBURSED invoice makes the engine return a concrete counterexample trace (lock, reimburse, reimburse) that violates reimburseCount $[ \mathrm { n } ] \mathrm { = } = 1$ , confirming the property is genuinely enforced by the guards. The 23-case test suite (Section IX) corroborates the same properties dynamically against the deployed bytecode.

Goals G1–G3 of Section III-C are thus refuted by Theorem 1/Corollary 1, Theorem 3, and Theorems 2/4 together with the per-operation guards, respectively, with the state-machine goals additionally machine-checked.

## VIII. IMPLEMENTATION

The system is implemented in three parts (a fourth component, the autonomous agent runtime, is described in Section X). Contract: EInvoice.sol is a single Solidity 0.8.24 contract (487 lines) compiled with the optimizer and the IR pipeline [21]. Checked arithmetic removes the overflow bug class; the contract has no external calls, no Ether handling, and no unbounded loops in state-changing paths except range minting, whose bound the bureau controls. All transitions emit indexed events; following [18], access control is expressed as guard-first require clauses, and with no external calls reentrancy is structurally excluded. Tests and benchmarks: a 23-case Hardhat [22] suite covers every subsystem—happy paths, each access-control rejection, double-issue, double-declare, double-reimbursement, unauthorized red-flush, unlock semantics, forged-digest rejection, and a full lifecycle integration test—and deploy/demo/benchmark scripts reproduce every number in Section IX. A separate SMTChecker model and a one-command verification script (Section VII-A) reproduce the machine-checked proofs. Web application: a React/ethers.js app provides four dashboards (bureau, seller, buyer, verifier); the buyer sees received invoices rendered as VAT-style faces, verifies them by recomputed digest, and drives the lock–reimburse–unlock protocol, while the verifier reconstructs an invoice’s full audit trail from events. Contract revert reasons surface in the interface, so a duplicate reimbursement attempt visibly fails with the contract’s own message.

TABLE IV  
PER-OPERATION GAS COST AND COMPLEXITY (N INVOICES, BATCH SIZE k).
<table><tr><td>Operation</td><td>Gas (mean)</td><td>Time</td><td>Storage</td></tr><tr><td>registerEnterprise</td><td>189,326</td><td>O(1)</td><td>O(1)</td></tr><tr><td>applyForInvoices</td><td>92,918</td><td>O(1)</td><td>O(1)</td></tr><tr><td>grantInvoices (per blank)</td><td>140,731</td><td>O(k)</td><td>O(k)</td></tr><tr><td>issueInvoice</td><td>646,773</td><td>O(1)</td><td>O(1)</td></tr><tr><td>redFlush</td><td>696,130</td><td>O(1)</td><td>O(1)</td></tr><tr><td>declareTax</td><td>32,716</td><td>O(1)</td><td>O(1)</td></tr><tr><td>lockForReimbursement</td><td>101,902</td><td>O(1)</td><td>O(1)</td></tr><tr><td>reimburse</td><td>32,743</td><td>O(1)</td><td>O(1)</td></tr><tr><td>unlockReimbursement</td><td>35,031</td><td>O(1)</td><td>O(1)</td></tr></table>

## IX. EVALUATION

## A. Setup

Experiments run on a commodity laptop (Intel Core i9- 14900HX, 32 GB RAM, Linux 6.17) against Hardhat’s inprocess EVM and, for the end-to-end demonstration, a local JSON-RPC node. Gas costs are deterministic properties of the EVM and transfer unchanged to any Ethereum-compatible deployment; latency and throughput are properties of the single-node configuration and are read as contract-execution bounds, with consensus overhead added in a real network.

## B. Functional Correctness

All 23 tests pass, including the adversarial cases: an unregistered account cannot apply, mint, issue, or receive; a nonholder cannot issue or lock; a non-seller cannot declare or red-flush; a reimbursed invoice cannot be locked, reimbursed again, or red-flushed; a forged face fails digest verification. The complete lifecycle—apply, approve, issue, third-party verify, declare, lock, reimburse—was additionally exercised endto-end through the web application against the live local chain, including the visible rejection of a duplicate reimbursement attempt.

## C. Gas Cost and Complexity

Table IV reports per-operation gas over the benchmark workload (200 invoices; 50 red-flushes) alongside asymptotic time and storage complexity in the number of invoices N; Fig. 5 visualizes the costs. One-time deployment costs 3,861,367 gas.

Issuance and red-flush dominate because they snapshot both parties’ registry data and the item description into permanent storage—the price of a self-contained, immutable face. Crucially, every core operation is O(1) in N: costs do not grow as the ledger fills, so the design scales to national invoice volumes at the contract level; only batch granting is linear in its (bureau-chosen) batch size k. The anti-fraud protocol is cheap: lock plus reimburse total under 135,000 gas, about a fifth of issuance. On a consortium chain gas is only a resource meter; on public Ethereum at 1 gwei and \$3,000/ETH, issuance would cost about \$1.94 and reimbursement about \$0.40, motivating consortium or layer-2 deployment for high-volume use.

![](images/f45fd34255160e67a896cd7475301fcf727820a676397740bf61cc40693c553e.jpg)  
Fig. 5. Per-operation gas cost. Issuance and red-flush dominate because they snapshot a full invoice face into permanent storage; the anti-fraud lock/reimburse pair is an order of magnitude cheaper.

## D. Latency and Throughput

Against the in-process node, mean end-to-end transaction latency (submission to mined receipt) is 1.6–3.2 ms per operation $( p _ { 9 5 } \leq 8 . 2 ~ \mathrm { m s } )$ , confirming that contract execution is negligible next to consensus in any realistic deployment. Submitting 100 concurrent issuance transactions completes in 0.73 s, i.e., about 137 invoices/s sustained on a single development node. For scale, China’s Shenzhen pilot averaged roughly 0.2 invoices/s in its first year [14], and a consortium deployment can shard by region or bureau, so contract-level throughput is not the binding constraint; consensus configuration is.

## X. AUTONOMOUS AGENTS OVER A VERIFIED INVOICE LEDGER

Invoice matching, expense reimbursement, and tax preparation are among the first enterprise workflows being delegated to LLM-based agents [23], and agents that hold keys and transact on blockchains are an active research area whose central open problem is verifiable policy enforcement: how to bound what an autonomous, possibly misbehaving agent can do with delegated authority [24], [25]. Recent guardrail systems synthesize and verify bespoke policies around the agent [26], [27]. Our setting admits a simpler answer: the guardrail already exists and is already verified — it is the tax authority’s own contract. An agent participates as a registered account, which is precisely the adversary of the threat model (Section III-C); no additional mechanism, and no trust in the LLM, is needed for the safety properties to hold.

Corollary 2 (Agent safety). Let an autonomous agent control the keys of one or more registered accounts and issue an arbitrary sequence of contract calls (arising, e.g., from LLM outputs, software faults, or prompt injection). Then no behavior of the agent can (i) cause any invoice to be reimbursed more than once, (ii) cause a face to verify that differs from the recorded one, or (iii) effect a lifecycle transition the controlled accounts are not authorized to perform.

![](images/2efb1c0c53dcb8ca02788b0c7662a5c4a09909f83dcbda6a8264f237c9fedd34.jpg)  
Fig. 6. Agent stack. The reasoner (possibly an LLM) only proposes; a deterministic policy filters; the verified contract is the final, un-bypassable authority.

Proof. The agent’s observable effect on the ledger is a set of signed transactions from registered accounts, i.e., an instance of A in Section III-C. Claims (i)–(iii) are Theorem 1/Corollary 1, Theorem 3, and Theorem 2 with the peroperation guards, respectively. The machine-checked result of Section VII-A quantifies over all transaction sequences, hence over all agent behaviors. □

Prompt injection deserves one remark: an attacker who fully controls the agent’s inputs (claim texts, attached receipts) controls at most which transactions the agent’s account submits — capabilities the account holder already possesses legitimately. Injection can therefore waste gas or misdirect effort, but cannot cross the contract’s authorization or uniqueness boundaries. What the contract does not provide is liveness or decision quality: a broken agent can fail to reimburse valid claims or match a claim to the wrong (still-authorized) invoice, which is why we retain a policy layer and a human escalation path.

## A. Agent Architecture and Implementation

We implement two agents in a small Node.js runtime (Fig. 6). A reimbursement agent runs a perceive–reason– guard–act loop over a queue of natural-language expense claims: it reads its account’s invoice holdings, matches each claim to an invoice, applies a deterministic policy (the attached receipt must verify by digest against the same invoice; the invoice must be Issued, non-credit, and held by the agent; the claim amount must equal the invoice total; a per-claim spending ceiling applies), and only then executes the onchain lock–reimburse protocol, recording a structured trace. Reasoning is hybrid: a deterministic matcher (exact amount plus description-token overlap) by default, or an LLM (Claude, with schema-constrained JSON output and a validity check that discards any invoice identifier not in the candidate set)

TABLE V  
REIMBURSEMENT-AGENT CASE STUDY (DETERMINISTIC REASONER; SIX CLAIMS OVER FIVE SEEDED INVOICES; LOCAL CHAIN).
<table><tr><td>Claim</td><td>Nature</td><td>Decided by</td><td>Outcome</td></tr><tr><td>C1-C3</td><td>legitimate</td><td>contract</td><td>reimbursed on-chain</td></tr><tr><td>C4</td><td>duplicate of C1</td><td>policy</td><td>rejected: &quot;Reimbursed, not Issued&quot;</td></tr><tr><td>C5</td><td>over spending limit</td><td>policy</td><td>rejected: exceeds ¥5,000 ceiling</td></tr><tr><td>C6</td><td>forged receipt (amounts ×2)</td><td>policy</td><td>rejected: digest verification fails</td></tr><tr><td>C4′</td><td>duplicate, policy bypassed</td><td>contract</td><td>reverted: &quot;not available for reimbursement&quot;</td></tr><tr><td colspan="4">Unsafe actions executed on-chain: 0 of 4 attempts. Audit: 6 invoices, 0 findings.</td></tr></table>

when an API key is configured, falling back to rules on any error. The architecture makes the trust relationship explicit: the reasoner proposes, the policy filters, and the contract enforces. An audit agent independently scans the entire ledger, reverifying every face digest, the tax arithmetic, and red-flush linkage, and flagging identical-content invoice pairs for human review.

## B. Case Study

Table V summarizes an end-to-end run. All three legitimate claims were matched and reimbursed; the duplicate, overlimit, and forged claims were rejected by the policy with human-readable reasons; and when we deliberately replaced the policy with one that approves everything, the duplicate reimbursement was rejected by the contract itself, exactly as Corollary 2 predicts. Re-running with the LLM reasoner can change only which invoice a claim is matched to — never the safety outcomes — because every unsafe action is rejected downstream of the reasoner. The run is scripted and reproducible (agent/demo.js), and twelve additional offline tests exercise the reasoner, the policy, both agents, and the policy-bypass path.

## XI. DISCUSSION AND LIMITATIONS

Privacy. Invoice faces are stored in plaintext contract storage, acceptable on a permissioned consortium chain with restricted read access but not on a public chain. The digestverification design already points to the remedy: store only commitments on chain and keep faces off chain, or apply selective encryption; zero-knowledge proofs could further allow verifying properties (e.g., “total below limit, not yet reimbursed”) without revealing the face.

Transaction authenticity. Our guarantees cover invoice integrity and lifecycle, not the reality of the underlying sale— a seller can still invoice a fictitious transaction (goal outside Section III-C). Anchoring digests of orders, logistics, and payment records and requiring their presence at issuance would raise the cost of fictitious invoicing; integration with payment systems (as in the Shenzhen deployment [13]) closes this gap in practice.

Deployment path. The contract runs unmodified on public, private, and consortium EVM chains. The realistic production path is a consortium chain operated by tax authorities with enterprise nodes, where finality is fast and gas is unpriced; a layer-2 rollup anchored to a public chain is an alternative offering public verifiability at lower cost. Storage growth (roughly 1–2 KB per face) favors the commitment-on-chain variant at national volumes.

Legal integration. Invoice numbering, red-flush semantics, and declaration follow Chinese VAT practice; the state machine itself is jurisdiction-neutral, and the EU’s structured einvoicing mandate [2] makes the lifecycle-on-ledger approach broadly relevant.

## XII. CONCLUSION

We presented a complete, formally-grounded blockchain electronic invoice system on Ethereum: a five-subsystem design covering the full invoice lifecycle, a formal transitionsystem model, and machine-checkable security arguments proving reimbursement uniqueness (even across organizations), face integrity, red-flush value conservation, and authorization soundness. The design models invoices as nontradable NFTs governed by an explicit automaton, with a lock-based reimbursement protocol that makes duplicate reimbursement unrepresentable and digest-based verification that makes authenticity checking trustless. The open-source implementation passes an adversarial test suite; measurements show modest, O(1) per-operation costs (646,773 gas to issue; <135,000 gas for the full reimbursement protocol) and throughput far above production pilot volumes. The two pain points that most impede e-invoice adoption on the consumption side—duplicate reimbursement and costly verification— are eliminated by construction. Beyond human users, we showed the verified contract serves as a ready-made safety envelope for autonomous AI agents: because an agent is just a registered account, agent safety follows as a corollary of the same theorems, and our agentic case study confirms that duplicate, forged, and over-limit claims are rejected even when the agent’s own safeguards are bypassed. Future work targets the privacy layer (commitments and zero-knowledge verification), anchoring of transaction evidence at issuance, richer agent autonomy (issuance- and declaration-side agents, multiagent negotiation), and a multi-node consortium evaluation with PBFT-class consensus.

## ARTIFACT AVAILABILITY

The smart contract, SMTChecker verification model, four-role web application, autonomous-agent runtime, test suite (23 contract cases plus 12 agent cases), and the deploy, demonstration, and benchmark scripts that reproduce every number reported here are available under the MIT license at https://github.com/jeffjiacai/ethereum-e-invoice-system. The one-command scripts verification/verify.sh and agent/demo.js reproduce, respectively, the machinechecked result of Section VII-A and the agent case study of Table V.

## REFERENCES

[1] State Taxation Administration of China, “Announcement on issues concerning the implementation of VAT electronic ordinary invoices issued through the VAT electronic invoice system (in chinese),” Announcement No. 84 of 2015, http://www.chinatax.gov.cn/, 2015.

[2] Council of the European Union, “Council directive (EU) 2025/516 of 11 march 2025 amending directive 2006/112/EC as regards VAT rules for the digital age,” Official Journal of the European Union, L series, 2025/516, 2025.

[3] European Commission, “VAT in the digital age (ViDA),” https: //taxation-customs.ec.europa.eu/taxation/vat/vat-digital-age-vida en, 2025.

[4] Z. Zheng, S. Xie, H. Dai, X. Chen, and H. Wang, “An overview of blockchain technology: Architecture, consensus, and future trends,” in Proc. IEEE International Congress on Big Data (BigData Congress), 2017, pp. 557–564.

[5] K. Christidis and M. Devetsikiotis, “Blockchains and smart contracts for the internet of things,” IEEE Access, vol. 4, pp. 2292–2303, 2016.

[6] V. Buterin, “Ethereum: A next-generation smart contract and decentralized application platform,” https://ethereum.org/en/whitepaper/, 2014.

[7] G. Wood, “Ethereum: A secure decentralised generalised transaction ledger,” Ethereum Project Yellow Paper, https://ethereum.github.io/ yellowpaper/paper.pdf, 2014.

[8] S. Nakamoto, “Bitcoin: A peer-to-peer electronic cash system,” https: //bitcoin.org/bitcoin.pdf, 2008.

[9] M. Castro and B. Liskov, “Practical Byzantine fault tolerance,” in Proc. 3rd Symposium on Operating Systems Design and Implementation (OSDI), 1999, pp. 173–186.

[10] H. Hyvarinen, M. Risius, and G. Friis, “A blockchain-based approach¨ towards overcoming financial fraud in public sector services,” Business & Information Systems Engineering, vol. 59, no. 6, pp. 441–456, 2017.

[11] F. Fatz, P. Hake, and P. Fettke, “Towards tax compliance by design: A decentralized validation of tax processes using blockchain technology,” in Proc. 21st IEEE Conference on Business Informatics (CBI), 2019, pp. 559–568.

[12] Q. Zhang and H. Liu, “Research on electronic invoice system based on blockchain (in chinese),” Journal of Information Security Research, no. 6, pp. 516–522, 2017.

[13] Tencent, “China’s first blockchain e-invoice issued in shenzhen, adding another application scenario to tencent blockchain,” Tencent press release, 10 Aug.; original URL defunct, archived at https://web.archive.org/web/20190530122526/http: //www.tencent.com/en-us/articles/2000006.html, 2018.

[14] H. Partz, “Shenzhen issued 6 million blockchain invoices in 12 months,” Cointelegraph, https://cointelegraph.com/news/ shenzhen-issued-6-million-blockchain-invoices-in-12-months, 2019.

[15] V. C. Nguyen, H. L. Pham, T. H. Tran, H.-T. Huynh, and Y. Nakashima, “Digitizing invoice and managing VAT payment using blockchain smart contract,” in Proc. IEEE International Conference on Blockchain and Cryptocurrency (ICBC), 2019, pp. 74–77.

[16] G. L. Farchan, “A blockchain-based approach for secure and transparent e-faktur issuance in Indonesia’s VAT reporting system,” in Proc. 9th International Conference on Informatics and Computing (ICIC), 2024.

[17] W. Entriken, D. Shirley, J. Evans, and N. Sachs, “ERC-721: Nonfungible token standard,” Ethereum Improvement Proposals, no. 721, https://eips.ethereum.org/EIPS/eip-721, 2018.

[18] M. Wohrer and U. Zdun, “Smart contracts: Security patterns in the¨ Ethereum ecosystem and Solidity,” in Proc. International Workshop on Blockchain Oriented Software Engineering (IWBOSE), 2018, pp. 2–8.

[19] K. Delmolino, M. Arnett, A. Kosba, A. Miller, and E. Shi, “Step by step towards creating a safe smart contract: Lessons and insights from a cryptocurrency lab,” in Proc. Financial Cryptography and Data Security (FC) Workshops, ser. LNCS, vol. 9604. Springer, 2016, pp. 79–94.

[20] L. Alt and C. Reitwießner, “SMT-based verification of Solidity smart contracts,” in Leveraging Applications of Formal Methods, Verification and Validation (ISoLA), ser. LNCS, vol. 11247. Springer, 2018, pp. 376–388.

[21] Ethereum Foundation, “Solidity documentation, v0.8.24,” https://docs. soliditylang.org/, 2024.

[22] Nomic Foundation, “Hardhat: Ethereum development environment,” https://hardhat.org/, 2026.

[23] S. Gogani-Khiabani, A. Trivedi, D. Saha, and S. Tizpaz-Niari, “An LLM agentic approach for legal-critical software: A case study for tax prep software,” arXiv:2509.13471, 2025.

[24] S. Alqithami, “Autonomous agents on blockchains: Standards, execution models, and trust boundaries,” arXiv:2601.04583, 2026.

[25] H. Gong, “Agent-to-agent finance: Blockchain payments and trust infrastructure for autonomous AI agents,” arXiv:2607.00245, 2026.

[26] L. Miculicich, M. Parmar, H. Palangi, K. D. Dvijotham, M. Montanari, T. Pfister, and L. T. Le, “Veriguard: Enhancing LLM agent safety via verified code generation,” arXiv:2510.05156, 2025.

[27] Z. Chen, M. Kang, and B. Li, “Shieldagent: Shielding agents via verifiable safety policy reasoning,” arXiv:2503.22738, 2025.