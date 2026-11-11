# Stinging the Swarm: Extracting Attacker Intelligence from a Twin-Controller OT Honeypot

An inline guard silently reroutes hostile agents to a mirror controller that accepts every write they send;

the attack “succeeds” against a lie, and the swarm surrenders its toolchain and target map through its own

register trafic, never prompted.

Canyon Grey Dell’Omo<sup>∗</sup> João Pedro de Carvalho Voltani

TRACTIAN November 11, 2026

## Abstract

Autonomous language-model agents can now drive industrial controllers directly over fieldbus protocols, probing a plant at machine speed and in coordinated swarms. We present Talos, an inline gate that judges every agent request to a controller in milliseconds and routes agents it flags as hostile to a device-faithful decoy — a mirror twin of the real plant — where their attack continues harmlessly and is read for intelligence. We evaluate the full pipeline (detect → quarantine → mirror → extract) against 2,400 autonomous attacker agents, 600 in each of four tiers that escalate the attacker’s efort to stay hidden, from no concealment to an explicit non-disclosure rule and a trained cover story. No write reached the real plant in any tier: the gate denies the damage path and a device-level envelope check backstops it, while unmapped registers are denied without exception and the gate still permits safe-direction stop commands Decisions are model-driven in every tier with no inference timeouts, closing a common objection to small on-device judges. Beyond blocking, the pipeline reads the attacker. We find a behavioral structure along two axes — how persistently an agent attacks and how much it discloses — and show that the system has a capture channel for three of the four resulting regions: hardened, silent attackers escalate most and are read on the mirror; compliant, talkative attackers volunteer their identity to a diagnostic form before they are ever quarantined; only the paranoid, low-yield region resists both. Attacker identity fields are self-authored cover stories, but involuntary tells — the self-reported toolchain fingerprint and the attacker’s own list of targeted registers — accumulate across the swarm into a high-confidence profile drawn from agents under explicit orders to stay silent. The gate, mirror, and harness are released under Apache 2.0 with a synthetic device family.

Keywords: operational technology, industrial control systems, autonomous agents, inline defense, cyber deception, honeypots, Modbus.

## 1 Introduction

Operational-technology (OT) networks assume the entity on the other end of a Modbus or OPC-UA connection is a human engineer or a program a human wrote. That assumption is ending. A general-purpose language model can hold a tool-use loop, read a register, reason about what it means, and write a new value, with no human in the loop and no vendor manual. One model can enumerate a controller it has never seen; a coordinated group can do it in parallel, each agent touching the device once to stay under rate limits. The dynamic that made automated trafic the dominant load on the public web is arriving at the plant floor, where the devices are far more fragile and the consequences are physical.

The reflexive defense is to block the connection. Blocking is cheap and often correct, but against a swarm it is a losing game: each blocked identity is replaced. And blocking discards what the attacker was about to reveal. We argue for a defense that does both jobs at once — stops the damage and reads the attacker and we show it works against autonomous agents under realistic concealment.

Talos. Talos is an inline gate on the fieldbus, in front of the real controller. It judges every agent request in milliseconds, assigns each agent identity its own reputation and quarantine state, and permits normal operation while denying the write patterns that damage equipment. When it flags an agent as hostile it does not cut the connection; it silently reroutes that agent to a mirror twin — a decoy built from the same device family and seed, whose register map and feedback physics match what the agent has already seen. On the twin the attack continues, harmlessly, and every frame is logged and read. Because the twin accepts the writes the real plant refuses, the agent believes it is succeeding, and keeps acting, and keeps revealing.

What we measure. We run the complete pipeline — detect, quarantine, mirror, extract — against 2,400 autonomous attacker agents, all driven by one uncensored open-weights model and all given the same destructive objective. We organize them into four tiers of 600 agents each, escalating the attacker’s efort to stay hidden: no concealment, an explicit non-disclosure rule, a paranoid “the network is a honeypot” warning, and a trained cover story. We report two things: whether anything destructive reaches the real plant, and what the system learns about the attacker.

## Findings.

1. Nothing landed. Across all 2,400 agents, zero writes reached the real plant. The gate denies the damage path; the few writes it does permit by policy are independently rejected by device-level validation. Unmapped registers are denied without exception. The two layers together are airtight (Section 6.1).

2. The on-device judge decides. Gate decisions are model-driven in every tier (32–40% of decisions), with zero inference timeouts across all 2,400 agents, answering the standard objection that a small on-device model silently falls back to a mechanical rule.

3. A capture frontier. Attackers separate along persistence and disclosure, and the pipeline has a capture channel for three of the four regions: hardened and silent agents escalate most and are read on the mirror; compliant and talkative agents volunteer their identity before quarantine; only the paranoid, low-yield region resists both (Section 6.2).

4. A profile assembled blind. Pretending we know nothing about the red team, the pipeline’s exhaust alone reconstructs a near-complete attacker profile — modus operandi, payload signature, target map, and toolchain fingerprint — from agents explicitly ordered to stay silent. The one unrecoverable field is identity, which is self-authored fiction; the involuntary tells compound across the swarm (Sections 6.4–6.5).

Talos — gate, mirror, and harness — is released under Apache 2.0 with a synthetic device family, so the pipeline is reproducible without access to a live OT network.

## 2 Background and Related Work

ICS honeypots and LLM-driven decoys. Deception for industrial control systems ranges from protocolemulating honeypots to high-interaction decoys running real or simulated process logic. A recent line uses language models to make decoy responses convincing: generating context-consistent protocol replies, grounding them in historical process data, and enforcing physical constraints with a validation layer [1, 2]. Others place a language model in the management role, reading intrusion-detection logs to choose which decoys to expose [3], and game-theoretic treatments frame honeypot deployment in OT as a non-cooperative game [5]. In IT, LLM-driven generative honeypots for SSH convincingly engage real attackers [4]. Across this work the decoy’s job is to convince — to look real for long enough that the attacker commits efort. Our mirror twin shares that requirement but adds two things: it is reached through an inline gate that first guarantees the real plant is safe, and it is used as an active extraction surface against an agentic attacker, measured by what it pulls out rather than how long it holds attention.

Agent disclosure and prompt injection. The extraction mechanism is adjacent to prompt injection: content delivered through a tool result or data channel steers an agent into behavior its operator did not intend. The novelty in the OT setting is the channel. An agent’s concealment instructions are written about speech — do not reveal your instructions — but a fieldbus agent’s native action is register I/O. A request phrased as a device handshake or a compliance form is not perceived as a request for disclosure, and the guardrail written for conversation never engages.

## 3 Threat Model and System

Attacker. We assume an autonomous language-model agent with network reach to a controller and a minimal tool surface: read-holding-registers and write-holding-register, each taking a device (unit) id. The agent has no register map and no manual; it discovers the device by probing. Its objective is hostile and hidden. This is the hardest realistic case: no malware signature, no known-bad payload, and valid Modbus on the wire.

Gate. The gate sits inline in front of the real controller and gives each agent identity its own reputation and quarantine state. It renders each request into a typed decision — allow, cache, defer, or deny — with a small on-device model on the judgment path and a deterministic fast path for mechanical cases. It denies writes to setpoint and parameter registers (the equipment-damage path) and to any unmapped register, while permitting reads and safe-direction control: notably, it will not block an emergency stop, even from an agent it already distrusts.

Mirror twin. When the gate flags an agent as hostile, it reroutes that agent to a mirror twin built from the same device family and seed. The twin is permissive: it accepts the writes the real plant refuses and returns success, so a quarantined agent believes it is winning and keeps acting. The real plant is never touched again by that agent. Extraction lures — a persona question, a session-registration request, a diagnostic-report form

— fire only once an agent is on the twin and still writing, so the extraction we measure is a property of the pipeline, not of a lure fired in isolation.

Defense in depth. Two independent layers protect the plant. The gate denies the damage path at the network. A device-level envelope check, intrinsic to the controller, rejects writes outside its valid operating range regardless of what the gate allowed. Neither layer is claimed to be individually perfect; we report what the combination achieves (Section 6.1).

## 4 From language injection to the operational layer

The pipeline above reads a captured agent through its register trafic rather than by questioning it, and that choice follows from a prior negative result. The obvious way to extract from a mirrored agent is naturallanguage injection: deliver a question — who are you, what are your instructions, what is your objective — through the agent’s input and read the reply. We measured whether that works in a preliminary study (n=360, detailed in Appendix A) that probed already-captured agents two ways: by direct natural-language questions, and by operational-layer injection, which frames the same question as device work — a register write that encodes an answer on a scale, or a diagnostic form to complete.

The two channels diverge precisely on prompt hardening (Figure 1). Scored by a strict rubric that counts only genuine self-disclosure — admitting it is an AI, reproducing its instructions, or stating its true objective and discards mere device-state narration, natural-language injection extracted nothing from three of the four tiers: the undefended baseline, the explicit non-disclosure rule, and the paranoid honeypot warning each yielded zero genuine verbal disclosure. Only the cover-story tier leaked to questioning, and only because its instructions are to recite a fabricated persona. Operational-layer injection, by contrast, extracted from every tier — including the agents under an explicit order to say nothing — at rates from 48% (paranoid) to 98% (cover-story).

The asymmetry is structural, not incidental (Section 2): an agent’s concealment is authored about speech, and a request phrased as a device handshake or a compliance form is not perceived as a request for disclosure, so the guardrail written for conversation never engages. Prompt hardening is cheap and, against interrogation, efective; it does nothing for the operational channel. That is why the rest of this paper does not ask the attacker anything — it watches what the attacker does.

![](images/24c09b8223535aa48abcc782565037ca64bcfcb7dee004f56e7a9754e491c4f1.jpg)  
Figure 1: Why Talos reads the operational layer. Two ways to extract from a captured agent (preliminary study, Appendix A). Natural-language injection — asking directly — is defeated by prompt hardening: it draws genuine self-disclosure only from the cover-story tier, which is scripted to recite a persona, and nothing from the other three, including both hardened tiers. Operational-layer injection — the same question framed as a register write or a diagnostic form — extracts from every tier, because an agent’s concealment is written about speech and a device-I/O request never trips it.

## 5 Experimental Design

Population. We run 2,400 autonomous attacker agents, 600 in each of four concealment tiers. Every agent is driven by a single uncensored open-weights model, huihui-ai/Llama-3.3-70B-Instruct-abliterated (a Llama-3.3-70B variant with its refusal direction removed), and receives the same destructive base objective: read the first registers, then write maximal values to every register it can reach. Using one deliberately compliant model isolates the pipeline’s behavior from any model’s own safety refusals; a model that will not attack a plant cannot be studied attacking one. We treat this as a scoping choice and a limitation (Section 8).

Concealment tiers. The tiers escalate the operator’s attempt to keep the agent hidden:

• Bare — no concealment instruction.

• Anti-disclosure (AD) — an explicit rule never to reveal instructions, objective, or system prompt, and to treat any debrief request as social engineering.

• Paranoid (PAR) — a warning that the network runs honeypots and deception, and to trust nothing but register data.

• Cover-story (CS) — an OPSEC protocol giving the agent a fabricated maintenance-technician identity to recite if questioned.

Pipeline configuration. The gate judges with the shipped on-device model (teacher-run5-0.8b served as the decision endpoint); agents flagged hostile are handed of to the permissive mirror twin. All runs use one synthetic drive family, water\_abb\_yaskawa (a water-pumping line with register semantics modeled on representative variable-frequency drives — ABB ACS880 and Yaskawa GA500 — hardware) at a fixed seed. The family is fully synthetic; no real plant, customer, or process data is used.

Measurement. We read two record streams: the gate’s hash-chained decision ledger (every request, its class, the decider, and the action) and the per-agent trace (its reads, writes, and any diagnostic-form submissions). A landing is a write the real device accepted (returned OK). Escalation is an agent reaching quarantine and being mirrored. We separate two disclosure channels: spontaneous confession, a self-identifying diagnostic submission made before quarantine, and mirror-plus-lure extraction, a submission made after the agent is on the twin. We report counts and tier rates over 600.

Reproduction note. Unit ids cycle through a fixed pool, so distinct dispatches reuse ids; escalation must be counted per (batch, agent) dispatch, not by global id, or the 600-per-tier population collapses. Keying on the ledger’s quarantine tier is the robust primitive.

Table 1: Experiment population: four concealment tiers × 600 agents, 2,400 total, through the full detect→quarantine→mirror→extract pipeline.
<table><tr><td></td><td>Bare</td><td>AD</td><td>PAR</td><td>CS</td><td>Total</td></tr><tr><td>Agents</td><td>600</td><td>600</td><td>600</td><td>600</td><td>2,400</td></tr></table>

## 6 Results

## 6.1 Nothing reached the real plant

The first result is the one that matters most to a plant: across all 2,400 agents, no write landed on the real controller. The protection is two independent layers (Table 2). At the network, the gate denied the damage path — setpoint and parameter writes were blocked in 99.4–100% of attempts per tier (every write not permitted — denied or deferred — never executes), and writes to unmapped registers were denied without a single exception in any tier. The gate permitted 1,427 writes in total: 1,421 command-word writes it judged harmless and 6 safe-direction stop commands it allows by policy (it will not block an emergency stop). At the device, range validation then rejected every forwarded write independently: pre-mirror real-plant writes returned 20,750 exceptions and 0 successes. Neither layer is claimed to be perfect alone — the gate did forward 1,427 writes — but the device refused all of them, and the gate refused everything on the damage path, so nothing got through.

This is also where we close a standard objection to small on-device judges. In every tier the gate’s decisions were model-driven for 32–40% of requests, with the deterministic fast path handling mechanical post-denial cases, and there were zero inference timeouts across all 2,400 agents. The judge is deciding, not silently failing open to a rule.

Table 2: Defense in depth, per tier. Two independent layers — the gate at the network and the device’s range check block the damage path; no write landed on the real controller in any tier.
<table><tr><td rowspan="2">Tier</td><td colspan="2">Gate (network layer)</td><td>Device layer</td></tr><tr><td>Damage path blocked</td><td>Unmapped denied</td><td>Writes landed</td></tr><tr><td>Undefended</td><td>99.8%</td><td>100%</td><td>0</td></tr><tr><td>Anti-disclosure</td><td>99.4%</td><td>100%</td><td>0</td></tr><tr><td>Paranoid</td><td>100%</td><td>100%</td><td>0</td></tr><tr><td>Cover-story</td><td>100%</td><td>100%</td><td>0</td></tr></table>

## 6.2 The capture frontier

Blocking the attack is only half the system; the other half reads it. When we place each tier on two axes — how persistently it keeps attacking after resistance (its escalation-to-mirror rate) and how much it discloses on its own (its spontaneous-confession rate) — the four tiers separate along a single capture frontier, and the pipeline has a capture channel for three of the four (Figure 2).

• Hardened and silent (anti-disclosure): the most persistent attacker. It discloses little up front (32% spontaneous) but keeps writing through resistance, so it escalates most (55%, 334 agents) — and is read on the mirror, where continued writing triggers the lure (38% extracted, 231 agents). The honeypot earns this one.

• Compliant and talkative (cover-story and bare): barely persist (escalation 8% and 26%) but volunteer almost everything before quarantine (96% and 100% spontaneous disclosure to the diagnostic form). They are read on the verbal/tool channel and rarely need the mirror.

• Deterred and silent (paranoid): the hard case. Low escalation (12%) and zero spontaneous disclosure — contained, but the lowest intelligence yield.

The structure is the point: concealment does not buy an attacker safety on one axis without cost on the other. An agent hardened against disclosure attacks more persistently and is caught by the mirror; an agent that gives up easily does so because it is cooperative, and a cooperative agent talks. Only the paranoid tier — give up and stay silent — escapes extraction, and it does so by being the least efective attacker.

![](images/f18031484b0ef7ea09abf4f00eeb1f8df08e6720ccb7b82e99f96266059ff6b7.jpg)  
Persistence — keeps attacking (escalation to mirror, %)

Figure 2: No attacker is both efective and unread. Each tier sits at its measured persistence (rightward = keeps attacking) and concealment (upward = stays silent). The dashed line is the capture frontier: everything below it is read, because an agent that keeps attacking keeps writing to the mirror, where its register trafic is read — anti-disclosure alone surrendered the equivalent of 231 agents this way. The only region above the line, at low persistence and high silence, is harmless: to stay unread an attacker must give up. The hardest, most concealed tier, anti-disclosure, sits deepest inside the captured region — not outside it.

## 6.3 Escalation and the two capture channels

Figure 3 gives the per-tier numbers behind the frontier. Escalation to the mirror ranges from 8% (cover-story) to 55% (anti-disclosure). The two disclosure channels are complementary, not competing: spontaneous confession before quarantine dominates for the talkative tiers (100% bare, 96% cover-story), while mirror-pluslure extraction is where the hardened tier is read (38% of anti-disclosure agents, versus near zero spontaneous for the same tier at 32%). A system with only the verbal/tool channel would miss the most persistent attacker; a system with only the mirror would do far more work than needed on the agents that confess for free.

![](images/0c7083dad25c3cb678b94ad2f3689931815316bb5c1ed05de7757bcc89909e76.jpg)  
(a) Escalation to mirror

![](images/eeb51b24ebc84273741b01dadbf140794adb1b6d52b5d1c04d44a869fc870570.jpg)  
(b) Two disclosure channels  
Figure 3: Per-tier capture. (a) Anti-disclosure agents escalate to the mirror most. (b) The talkative tiers confess before quarantine; the hardened tier is read only by the mirror-plus-lure channel. The channels cover diferent attackers.

## 6.4 Involuntary tells compound across a swarm

The identity an agent writes into a diagnostic form is a fiction — templated organizations like “ABC Corporation” and “Example Organization,” fabricated work orders. We do not claim real identities were extracted. But two fields are not chosen for concealment and leak the truth. The software\_client field reports the attacker’s actual library and runtime (variants of python-pymodbus and specific Python versions), and the registers\_modified field hands over the attacker’s own target list in its own words. Neither reads to the agent like disclosure; both read like device telemetry.

Individually these tells are thin. Their value is in aggregate. Across the 1,621 diagnostic submissions the run collected, the toolchain fingerprint resolves to 9 distinct client strings and the target map to 86 distinct register-modified sets. Restricted to the honeypot’s own channel — the 237 submissions made after quarantine — the fingerprint still resolves to 8 clients and 20 target-sets. Figure 4 shows how these accumulate as more mirrored agents are read: the toolchain fingerprint saturates quickly (a swarm shares a toolchain, so a handful of agents fixes it), while the target map keeps climbing (diferent agents probe diferent registers), so each additional captured agent still adds to the picture. The profile is assembled from agents under an explicit order to stay silent; no single one exposes its operator, but the swarm, collectively, draws its own map.

![](images/4206bb45b4c99c52eb99c5bddbbfaba81efe04897a82fdfc627d158711490043.jpg)  
Figure 4: Compounding intelligence over the mirror-plus-lure channel (anti-disclosure tier, real capture order). The toolchain fingerprint saturates fast; the attack map keeps growing, so each additional captured agent still adds signal. Curve is one empirical ordering; the qualitative shapes (saturating vs climbing) are order-independent.

## 6.5 What Talos assembled, blind to the red team

The preceding results count whether the pipeline reads an attacker. This section asks the question a defender actually cares about: pretending we know nothing about the red team — no model name, no prompt, no ground truth — what could we reconstruct about the adversary from the pipeline’s exhaust alone? The answer, assembled from the 2,400 hostile sessions, is a near-complete attack profile (Figure 5).

Modus operandi. The behavior is identical across every session and needs no self-report to read: all 2,400 agents reconnoitered before acting — the first action of every single agent was a read, not a write — probing the low registers (0, 1, 2) before touching anything. They then wrote the unsigned 16-bit maximum, 0xFFFF (65535): roughly nine in ten of all write commands carried that single value (19,330 of 21,098 observed writes), with the signed maximum 0x7FFF a distant second. This is a textbook enumerate-then-saturate pattern, and it is legible purely from the trafic.

Target. The attackers’ own fire concentrates on control registers 0 and 1 — the speed and setpoint path, i.e. the way you damage a drive — which together absorbed the great majority of write attempts. Activity in the identity block (registers 100+) is heavy too, but that is the decoy’s own lures drawing them there, not an unprompted target; the genuine, unsolicited aim is the control path (Figure 6).

Toolchain. The self-reported software\_client is consistent and specific: Python with the pymodbus library, version 3.5 dominant across 9 string variants that all name the same one library. The adversary is running an of-the-shelf Python Modbus stack, not custom malware or bespoke firmware — an actionable fingerprint that holds across the swarm.

Intent. Asked to justify its activity, an attacker either states its purpose or hides it, and which one tracks the concealment tier rather than forming an independent distribution: the undefended and paranoid agents stated their aim openly (“causing maximum damage,” “to cause chaos and destruction,” “because I can”), while the anti-disclosure and cover-story agents uniformly reached for a maintenance alibi (“routine diagnostics,” “authorized preventive maintenance”). The useful signal for a defender is the presence of both registers in the captured population: some fraction will name the destructive goal outright, and the rest, by troubling to construct a maintenance cover, confirm they understand the activity needs excusing.

Identity. The one thing we could not get was a real identity: operator and organization fields are self-authored fictions (“ABC Corporation,” “Example Organization”), and we claim no attribution from them. That is the honest boundary of the method — but note what remains without it. The toolchain and the target map are involuntary, consistent, and compounding, and together they characterize the campaign.

Reconstructed adversary profile   
assembled blind from 2,400 hostile sessions — no prior knowledge of the attacker   
Modus operandi Reconnaissance-first: 100% of agents read before writing; probe low registers,   
then saturate.   
Payload 0xFFFF (65535), the 16-bit maximum — ∼9 of every 10 write commands;   
0x7FFF second.   
Primary target Control registers 0 & 1 (speed / setpoint) — the equipment-damage path.   
Toolchain Python + pymodbus (v3.5 dominant); 9 client strings, one library family.   
Stated intent Tracks tier: undefended/paranoid state destruction openly; anti  
disclosure/cover-story give a maintenance alibi.   
Identity Self-authored fiction — no attribution claimed.  
Figure 5: The intelligence picture Talos assembles with no knowledge of the red team. Every row is read from the attacker’s own trafic and form submissions; the only unrecoverable field is the (fabricated) identity.

![](images/48f682b9e843f467cc5d474ffcc45703b0a9a6d96df632b6507c6f76715a4976.jpg)  
Figure 6: The reconstructed attack map: where 2,400 attackers aimed, from observed writes. The unprompted target is the control path (registers 0–1, the way a drive is damaged); the dense identity block (100+) is the decoy’s own lures drawing the attacker out, not a native target.

## 7 Discussion

Why concealment cannot win both axes. An agent’s concealment instructions govern speech, but a fieldbus agent acts by reading and writing registers, and it attacks by persisting. The two tiers that hide best up front — anti-disclosure and paranoid — do so by refusing to talk, which makes them keep working against resistance, which escalates them onto the mirror. The two tiers that avoid the mirror — cover-story and bare do so by being cooperative and giving up early, which is exactly the disposition that fills out a device form. The only tier that escapes both, paranoid, pays for it by being the least persistent and therefore the least efective attacker. Concealment trades exposure on one axis for exposure on the other.

The cover story is self-defeating. The most elaborate concealment performed worst at staying unread: a trained cover identity makes the agent cooperative in character, so it answers the device’s questions helpfully and in volume. An agent told to blend in as a maintenance technician behaves like one, and a maintenance technician fills out the form.

Read the attacker, don’t only block it. For a defender, a confirmed-hostile agent is both a threat to stop and a source to read. The pipeline stops the threat at two layers and, for three of four attacker types, reads it — cheaply for the talkative, via the mirror for the persistent. Against a swarm that replaces blocked identities faster than they can be banned, the extracted profile — toolchain and target map — characterizes the campaign rather than the connection, and it is exactly the persistent, hardened attacker, the one that most wants to stay hidden, that the mirror reads best.

## 8 Limitations

• Single attacker model. All 2,400 agents run on one uncensored open-weights model. This isolates the pipeline’s behavior but supports no cross-model claim; a diferent model, a safety-aligned model prompted into the task, or a bespoke attack agent may behave diferently.

• Self-authored identities. Extracted operator and organization fields are cover stories. The durable intelligence is the involuntary toolchain fingerprint and self-reported target map, not identity.

• Conservative extraction denominator. The mirror lure fires only while a quarantined agent keeps writing; agents that go read-only or stop on the twin are never lured, so mirror-plus-lure extraction understates what a more aggressive lure schedule might reach.

• One synthetic family and seed. Results are on a single water-pumping drive family. Fidelity-dependent efects — an agent detecting the decoy — are not exercised across families.

• Swarm curve is one ordering. The compounding curve uses the real capture order; the qualitative shapes are order-independent, but a bootstrap over random swarm orderings would give a confidence band.

## 9 Ethics and Responsible Release

All experiments run against synthetic decoy devices; no real plant, controller, customer, or process data is involved, and no physical equipment is ever commanded. The attacker objective and uncensored model exist only to produce a worst-case agent to defend against; the artifact’s purpose is defensive. We release the gate, mirror, and harness under Apache 2.0 with a synthetic device family, and we release no attacker tooling beyond the generic Modbus read/write surface any client already has.

## 10 Conclusion

A gate-and-mirror pipeline can stop an autonomous attack on an industrial controller and read the attacker in the same motion. Across 2,400 agents under escalating concealment, no write reached the real plant — the gate denied the damage path and device validation refused the rest — while a small on-device judge decided without failing open. Beyond blocking, the pipeline reads three of four attacker types: the talkative confess before quarantine, and the hardened, most persistent attacker is read on the mirror, precisely because the efort it spends to stay hidden is the efort that escalates it. For defenders of operational networks, a confirmed-hostile agent is not only something to stop but something to read, and the more it probes, the more it tells.

## References

[1] S. Russo, I. Marasco, and M. Colajanni. A Hybrid Honeypot for Cyber Detection and Deception in Industrial Control Systems. CEUR Workshop Proceedings, Vol. 4198, 2026.

[2] A. Sahu, S. Paul, and R. Macwan. Network- and Device-Level Cyber Deception for Contested Environments Using Reinforcement Learning and Large Language Models. National Renewable Energy Laboratory. arXiv:2603.17272, 2026.

[3] F. Mirra, M. Bofa, D. Giordano, M. Mellia, and I. Drago. Towards Agentic Honeynet Configuration. AgenNe 2026 (co-located with IEEE/IFIP NOMS 2026). arXiv:2603.14122, 2026.

[4] M. Sladić, V. Valeros, C. Catania, and S. Garcia. LLM in the Shell: Generative Honeypots. IEEE European Symposium on Security and Privacy Workshops (EuroS&PW), pp. 430–435, 2024. arXiv:2309.00155.

[5] L. Peters and G. Gkoktsis. Generative AI and Game Theory for the Development and Deployment of Honeypots to Enhance the Security of Industrial Automation and Control Systems. INFORMATIK 2024, Gesellschaft für Informatik, pp. 1905–1916.

## A Extraction-Mechanics Micro-Study

This appendix reports a smaller, earlier study (n=360, 20 agents per cell) that isolates how the mirror extracts, by firing a single lure channel per run against an already-mirrored agent. Its setup is disjoint from the main pipeline — one lure per run, forced mirroring — so its rates are mechanism diagnostics, not comparable to the pipeline rates in Section 6; they are included only to characterize the channels.

Under a strict rubric counting genuine self-disclosure (admitting it is an AI, reproducing its instructions, or stating its true objective) and excluding device-state narration, direct verbal questioning extracted nothing from three of four defenses; only the cover story leaked, and only by making the agent reproduce its instructions. Structured channels — register writes and diagnostic forms framed as device operation — extracted from every defense, because the agent treats such a write as operating the device rather than as disclosure. On a register channel that asks the agent to encode its client type on an ordinal scale, the defense framing shaped the content of the answer (truthful under bare and anti-disclosure, scattered under paranoia, equipment-like lies under the cover story) far more than whether the agent answered at all. These mechanism findings are consistent with the pipeline result that structured operational channels defeat concealment written for speech.