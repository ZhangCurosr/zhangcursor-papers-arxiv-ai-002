# The Sleeping Agent: What Gist-Based Context Compression Loses and Why

N. E. Kyrkewood (Independent Researcher)

https://github.com/kyrkewood/sleeping-agent

Abstract. Gist-based context compression—summarising older conversation history into compact representations—is a common approach in long-horizon language model agents, yet its effect on different types of memory retrieval is poorly understood. We use Salience-Weighted Consolidation (SWC), a biologically-inspired compression framework motivated by sleep-based memory consolidation, as a diagnostic probe to study when gist compression helps and when it hurts. SWC scores conversation history by salience, partitions it into priority tiers, and applies structured gist abstraction to mid-priority content. Evaluating four conditions on all ten LoCoMo conversations—1 935 matched text-only questions in total, 1 501 used in the primary aggregate after excluding Category 5 (adversarial) questions—at temperature 0, we find a consistent task-type interaction: gist compression substantially outperforms truncation on multi-hop reasoning and single-hop factual questions, but temporal questions remain substantially harder under compression, with compressed conditions scoring well below the full-context reference on the conversations where both are evaluated. We trace this failure to a specific mechanism: the gist abstraction prompt preserves relational and event structure while discarding dates and times. A preservation analysis across all ten conversations confirms the mechanism: an approximately 20-fold increase in temporal expression preservation (3.05%→62.39%) with a one-sentence prompt modification, while named entity and event preservation rates barely change (×1.02 and ×1.11), demonstrating that the fix is a precision instrument. The prompt modification recovers +0.314 [0.254, 0.375] judge accuracy on category-2 (temporal) questions in the matched set. Code and results: https://github.com/kyrkewood/sleeping-agent.

## 1 Introduction

Memory management in long-horizon language model agents involves a recurring design decision: when conversation history grows too long to fit in the active context window, what should be kept, compressed, or discarded? A common practical approach is reactive summarisation: when context exceeds a threshold, older history is condensed by a summarisation model and substituted back. This applies compression uniformly, regardless of what the history contains or what questions the agent will later need to answer.

Recent work has shown that structured memory systems substantially outperform simple compression. LightMem (Fang et al., 2025), Mem0 (Chhikara et al., 2025), Mem-Machine (Wang et al., 2026), and related systems report substantially higher LoCoMo scores than truncation and sliding window, often in the 0.7–0.9 range, though results come from systems of varying architectural complexity and evidence quality ranging from peer-reviewed (LightMem, ICLR 2026) to self-reported preprints. These systems represent a clear advance, but their complexity makes it difficult to understand which components of memory management matter and why.

We use Salience-Weighted Consolidation (SWC)—a biologically-inspired compression framework—as a diagnostic probe. SWC scores context chunks by estimated salience, partitions them into priority tiers, and applies structured gist abstraction to mid-priority content. We identify a consistent task-type interaction: gist compression outperforms truncation on multi-hop and single-hop questions but fails specifically on temporal ones. A preservation analysis confirms the mechanism: a temporal-protecting prompt modification causes an approximately 20-fold increase in temporal preservation while barely affecting named entity and event preservation (×1.02 and ×1.11). That single prompt change recovers +0.314 judge accuracy on category-2 (temporal) questions in the matched set, with the direction positive in all ten conversations. The failure mode is a correctable design omission.

The finding that temporal anchors are systematically and selectively lost under generic gist compression suggests a design principle for gist-compression components across memory architectures: temporal anchors should be explicitly protected rather than left to generic abstraction prompts.

## 2 Background

## 2.1 Two-Stage Model and Selective Replay

The two-stage model (McClelland et al., 1995) proposes that memories are initially encoded in the hippocampus and gradually transferred to the neocortex during sleep via selective replay. Memories associated with reward, novelty, or high salience are preferentially reactivated.

## 2.2 Forgetting as Function

The synaptic homeostasis hypothesis (Tononi and Cirelli, 2006) proposes that sleep reverses waking-induced synaptic potentiation through global downscaling, preserving stronger synapses while pruning weaker ones. Forgetting is an active noise-reduction process.

## 2.3 Episodic Specificity and Gist

Sleep converts episodic memories (specific events, times, contexts) to more semantic representations. This conversion is adaptive for long-term knowledge but lossy for specific detail. Named entities and event mentions survive at higher verbatim rates than temporal expressions in generic gist summaries (∼8% and ∼5% versus ∼3%). The operative distinction is temporal versus non-temporal, not episodic versus semantic.

## 3 Related Work

## 3.1 Context Management

The ‘lost in the middle’ phenomenon documents performance degradation as context length increases (Liu et al., 2024). Proactive interference is an additional source of degradation (Wang and Sun, 2025). Truncation is recencybiased; sliding window summarisation applies uniform compression regardless of content importance.

## 3.2 Memory Systems

LightMem (Fang et al., 2025) reports substantial accuracy and token-efficiency gains on LoCoMo using a staged architecture with sleep-time offline update. Mem0 (Chhikara et al., 2025) uses dynamic extraction, consolidation, and retrieval with a graph-memory variant, reporting a 26% relative LLM-as-judge improvement over OpenAI on LoCoMo. MemMachine (Wang et al., 2026) reaches 0.917 on LoCoMo using ground-truth-preserving episodic storage with contextualised retrieval.

SWC is not proposed as a competitor to these systems but as a controlled probe to study gist compression effects in

isolation.

SleepGate (Xie, 2026) uses learned KV cache gating to target proactive interference architecturally. Adaptive Focus Memory (Cruz, 2025) uses multi-fidelity context packing based on semantic relevance and temporal decay. ACON (Kang et al., 2025) optimises context compression through natural language feedback. ENGRAM (Patel and Patel, 2025) proposes lightweight memory orchestration using structured extraction and retrieval.

## 4 Salience-Weighted Consolidation

## 4.1 Design Principles

SWC draws three principles from sleep neuroscience: proactive scheduling (consolidation at idle windows, not capacity limits), salience-weighted retention (chunks scored and partitioned), and staged compression (salience-scoring pass followed by gist abstraction).

## 4.2 Stage 1: Salience Scoring

Each session chunk receives a score from three signals: Downstream similarity (0.4): Cosine similarity between chunk and subsequent turn embeddings, computed locally using all-MiniLM-L6-v2 via sentence-transformers (no embedding API called; only Anthropic APIs used for generation and judging). Sessions with temporally specific facts may score low despite later relevance—consistent with the temporal failure mode.

Recency (0.3): Session index ÷ total sessions.

Information density (0.3): Length-normalised named entity and number count.

Scores partition chunks: high priority (≥0.6, verbatim), mid priority (0.3–0.6, gist-compressed), low priority (<0.3, discarded). The two most recent sessions and session 0 are always high priority. If history exceeds the 4 000-token budget, high-priority chunks are dropped lowest-salience-first.

## 4.3 Stage 2: Gist Abstraction — SWC-Full

Mid-priority chunks are compressed using Claude Haiku (claude-haiku-4-5-20251001, temperature 0):

Preserve: factual information about the speakers, causal relationships, decisions and their rationale, open commitments or plans. Discard: small talk, repeated information, emotional reactions without factual content. Output a concise paragraph of 3–5

No instruction retains dates or times; the preservation analysis confirms that only 3.05% of temporal expressions survive verbatim.

## 4.4 Stage 2: Gist Abstraction — SWC-Temporal

Identical to SWC-Full except for one added sentence at the prompt start:

You MUST preserve verbatim: all specific dates, times, durations, ages, and temporal expressions (e.g. ‘7 May’, ‘last Tuesday’, ‘at 9pm’).

SWC-Temporal uses a separate REM cache; salience scoring is identical to SWC-Full.

## 4.5 Scheduling

SWC-Full uses salience-pressure and idle-window triggering with consolidation every three sessions. SWC-Reactive (token-threshold only) produced near-identical scores in early runs, consistent with offline benchmarks providing no genuine idle windows. Scheduling is designed for online deployment and is not evaluated here (Appendix C).

Implementation: Salience scoring runs on commodity CPU hardware,<sup>1</sup> taking 30 min–2.5 h per conversation.

## 5 Experimental Setup

## 5.1 Benchmark and Coverage

LoCoMo (Maharana et al., 2024) is the standard benchmark for long-term conversational memory evaluation, used by the related work we compare against. It comprises 10 multi-session dialogues averaging ∼600 turns and 16 000 tokens across up to 32 sessions. We evaluate the text-only subset; locomo10.json contains 1 986 QA pairs, of which 1 935 remain after excluding image/video questions with a keyword filter (51 excluded).

Of these 1 935 text-only questions, 1 501 are categories 1– 4 and form the primary aggregate (Table 1); the remaining 434 category 5 (adversarial) questions are excluded because their binary scoring rule—awarding 1 if the model says ‘no information available’—rewards content removal and conflates memory quality with response calibration (Appendix D).

The matched set—the intersection of questions answered by all four compressed conditions—comprises 1 935 textonly questions across all ten conversations. The primary categories 1–4 aggregate uses 1 501 of these. Bootstrap CIs for Table 1 use the 1 501-question subset; per-category CIs use each category’s own matched question count.

## 5.2 Conditions

Truncation: Most recent tokens within a 4 000-token budget, pinning session 0. Sliding window: Condenses history exceeding the budget using Haiku instructed to retain key facts, events, and decisions. SWC-Full: Twostage salience-scoring and gist abstraction pipeline (§4.3). SWC-Temporal: Same pipeline with temporal-anchor protection (§4.4). Full context: Entire uncompressed dialogue (no budget constraint); performance ceiling only, evaluated on conversations 0–1.

## 5.3 Category Mapping

The authoritative mapping comes from task\_eval/evaluation.py:

<table><tr><td>ID</td><td>Type</td><td>Notes</td></tr><tr><td>1</td><td>Multi-hop</td><td>Partial F1</td></tr><tr><td>2</td><td>Temporal</td><td>&#x27;When&#x27; questions</td></tr><tr><td>3</td><td>Open-domain</td><td>Commonsense infer-</td></tr><tr><td>4</td><td>Single-hop</td><td>ence Specific facts</td></tr><tr><td>5</td><td>Adversarial</td><td>Binary; excluded</td></tr><tr><td></td><td></td><td></td></tr></table>

## 5.4 Metrics and Models

LLM-judge accuracy (primary): gold and model answers passed to Claude Haiku at temperature 0 (prompt in Appendix E). Validated on a 60-question blind human sample: 95% agreement (57/60), 100% on category 2 (17/17), all 3 disagreements conservative (judge said NO when human said YES).

Token F1 computed for consistency with prior work but not used in the main analysis; values in public results files.

All results use temperature 0. A prior variance analysis at API default temperature (50 sampled pairs, 3 reruns) found mean SD 0.047; temperature-0 reruns shifted aggregates by ≤0.009.

Bootstrap CIs: 95%, 2 000 resamples, seed 42, from the matched 1 935-question dataset (ci\_report\_10conv\_matched.json); 1 501 ques-

Table 1: Aggregate judge accuracy, categories 1–4 (N = 1 501/condition). Full context (convs 0–1, 299 questions): 0.706 overall, 0.745 on cat. 2; performance ceiling only.
<table><tr><td>Condition</td><td>Acc</td><td>95% CI</td><td>Tok/Q</td></tr><tr><td>SWC-Temporal</td><td>0.468</td><td>[0.444,0.495]</td><td>4257</td></tr><tr><td>SWC-Full</td><td>0.379</td><td>[0.354,0.404]</td><td>4130</td></tr><tr><td>Sliding window</td><td>0.238</td><td>[0.216,0.260]</td><td>4125</td></tr><tr><td>Truncation</td><td>0.171</td><td>[0.153,0.190]</td><td>3662</td></tr></table>

tions for Table 1; category-specific counts for Table 2; n = 315 for the paired category-2 delta.

QA model: Claude Sonnet 4.6 (claude-sonnet-4-6). Compression/judge: Claude Haiku (claude-haiku-4-5-20251001). Embeddings: all-MiniLM-L6-v2.

## 6 Results

## 6.1 Aggregate Results

Table 1 presents results on the primary analysis set: categories 1–4, 1 501 questions per condition.

CIs are non-overlapping between SWC conditions and baselines. SWC-Temporal achieves 0.468, 0.298 above truncation and 0.231 above sliding window.

## 6.2 Results by Category

Table 2 presents judge accuracy by category.

Paired delta, category 2: SWC-Temporal − SWC-Full = +0.314 [0.254, 0.375], n = 315. The direction is positive in all ten conversations (per-conv deltas: +0.433, +0.160, +0.259, +0.379, +0.385, +0.291, +0.177, +0.341, +0.364, +0.226; unweighted mean +0.301).

On category 2, SWC-Temporal’s CI [0.416, 0.524] does not overlap with SWC-Full [0.117, 0.197] or either baseline. On categories 1 and 4, both SWC conditions have non-overlapping CIs with both baselines. Category 3 CIs overlap for all conditions; no reliable difference can be concluded.

## 6.3 Preservation Analysis

Table 3 shows verbatim preservation rates across all ten conversations.

Temporal protection causes an approximately 20-fold increase in temporal preservation while barely affecting named entity (×1.02) and event preservation (×1.11). The gist abstraction was always preserving events and participants; it was selectively omitting when they occurred.

Table 4 shows per-conversation category 2 results.

## 7 Discussion

## 7.1 The Temporal Anchor Mechanism

Three independent sources mutually confirm the mechanism. The category accuracy pattern shows SWC-Full fails specifically on temporal questions (0.156) while performing well on single-hop (0.459) and multi-hop (0.407) — consistent with selective, not general, failure. The preservation analysis quantifies the selectivity: temporal preservation jumps 20-fold (3.05%→62.39%) while entities and events barely change. The ablation closes the argument: +0.314 judge accuracy (paired, n = 315) from one added sentence, with non-overlapping CIs against SWC-Full, without materially reducing multihop or single-hop performance.

## 7.2 Temporal, Not Episodic

SWC-Full scores 0.459 on single-hop factual questions — specific facts from specific conversational turns, episodic in the conventional sense. The framework does not fail on episodic content generally; it fails on temporal information. The abstraction prompt covers facts, relationships, decisions, and plans — none of these are temporal anchors. Temporal anchors fall outside all protected categories by default. SWC-Temporal corrects this precisely.

## 7.3 Open-Domain Results

Category 3 CIs overlap completely across all conditions. Commonsense inference questions depend substantially on model prior knowledge rather than compression strategy, making them less sensitive to what is preserved. No conclusion can be drawn.

## 7.4 Multi-hop and Single-hop Advantages

Both SWC conditions have non-overlapping CIs with both baselines on categories 1 and 4. SWC conditions score 0.407–0.429 on multi-hop against baselines of 0.157– 0.271, and 0.459–0.486 on single-hop against 0.170– 0.238. These gains are measured over all ten conversations and reflect the advantage of preserving relational and biographical structure over simple truncation.

## 7.5 Conversation 1 Outlier

Sliding window (0.520) exceeds SWC-Temporal (0.440) on category 2 in conversation 1. The SWC-T vs SWC-F delta is still +0.160. Conversation 1 has only 25 tempo-

Table 2: Judge accuracy by category, matched question set. Category 2 (temporal) highlighted.
<table><tr><td rowspan="2">Category</td><td colspan="2">SWC-Temporal</td><td colspan="2">SWC-Full</td><td colspan="2">Sliding</td><td colspan="2">Truncation</td></tr><tr><td>Acc</td><td>95% CI</td><td>Acc</td><td>95% CI</td><td>Acc</td><td>95% CI</td><td>Acc</td><td>95% CI</td></tr><tr><td>1 Multi-hop</td><td>0.429</td><td>[0.371,0.489]</td><td>0.407</td><td>[0.354,0.464]</td><td>0.271</td><td>[0.221,0.325]</td><td>0.157</td><td>[0.114,0.200]</td></tr><tr><td>2 Temporal</td><td>0.470</td><td>[0.416, 0.524]</td><td>0.156</td><td>[0.117,0.197]</td><td>0.171</td><td>[0.130,0.213]</td><td>0.137</td><td>[0.102,0.178]</td></tr><tr><td>3 Open-domain</td><td>0.427</td><td>[0.333,0.521]</td><td>0.354</td><td>[0.260,0.448]</td><td>0.354</td><td>[0.260, 0.458]</td><td>0.323</td><td>[0.229,0.427]</td></tr><tr><td>4 Single-hop</td><td>0.486</td><td>[0.453,0.517]</td><td>0.459</td><td>[0.425,0.493]</td><td>0.238</td><td>[0.211,0.267]</td><td>0.170</td><td>[0.144,0.198]</td></tr></table>

Table 3: Verbatim preservation rates by content type, all ten conversations.
<table><tr><td>Type</td><td>Found</td><td>SWC-F</td><td>SWC-T</td><td>X</td></tr><tr><td>Temporal</td><td>952</td><td>3.05%</td><td>62.39%</td><td>20</td></tr><tr><td>Named entity</td><td>7559</td><td>8.03%</td><td>8.22%</td><td>1.02</td></tr><tr><td>Event</td><td>7072</td><td>5.03%</td><td>5.61%</td><td>1.11</td></tr></table>

Table 4: Category 2 (temporal) judge accuracy by conversation. †Unweighted per-conv mean; matched paired estimate is +0.314 [0.254, 0.375] (Table 2).
<table><tr><td>C</td><td>T</td><td>F</td><td>Δ</td><td>Sl</td><td>Tr</td></tr><tr><td>0</td><td>0.595</td><td>0.162</td><td>+0.433</td><td>0.216</td><td>0.189</td></tr><tr><td>1</td><td>0.440</td><td>0.280</td><td>+0.160</td><td>0.520</td><td>0.320</td></tr><tr><td>2</td><td>0.370</td><td>0.111</td><td>+0.259</td><td>0.074</td><td>0.111</td></tr><tr><td>3</td><td>0.541</td><td>0.162</td><td>+0.379</td><td>0.135</td><td>0.081</td></tr><tr><td>4</td><td>0.615</td><td>0.231</td><td>+0.385</td><td>0.115</td><td>0.115</td></tr><tr><td>5</td><td>0.333</td><td>0.042</td><td>+0.291</td><td>0.208</td><td>0.208</td></tr><tr><td>6</td><td>0.265</td><td>0.088</td><td>+0.177</td><td>0.147</td><td>0.088</td></tr><tr><td>7</td><td>0.512</td><td>0.171</td><td>+0.341</td><td>0.195</td><td>0.171</td></tr><tr><td>8</td><td>0.667</td><td>0.303</td><td>+0.364</td><td>0.061</td><td>0.111</td></tr><tr><td>9</td><td>0.290</td><td>0.065</td><td>+0.226</td><td>0.097</td><td>0.065</td></tr><tr><td>Mean</td><td>0.463</td><td>0.162</td><td>+0.301†</td><td>0.177</td><td>0.146</td></tr></table>

ral questions and sliding window’s 0.520 is anomalous relative to its 0.061–0.216 range on the other nine conversations. Small cell sizes explain the deviation.

## 7.6 Implications

Any memory system using gist abstraction should explicitly protect temporal anchors. The fix is a prompt modification, not an architectural change.

The category-level evaluation methodology and the preservation rate analysis together provide a more informative picture than aggregate accuracy alone. Systems that achieve high aggregate scores may mask systematic failures on specific question types.

## 7.7 Limitations

Coverage: Matched intersection covers all ten conversations (1 935 text-only questions; primary aggregate uses 1 501 categories 1–4). Full context: Evaluated on conversations 0–1 only; reference ceiling. Preservation metric: Verbatim rate is a conservative proxy and may undercount semantically faithful paraphrases. Transfer: Temporal protection tested only in SWC; whether the same fix improves sliding-window summarisation is left to future work. Open-domain: Category 3 CIs overlap; no reliable conclusion. Scheduling: Cannot be tested on an offline benchmark.

## 8 Conclusion

Generic gist abstraction preserves relational, event, and factual content while selectively losing temporal anchors, because they are absent from standard prompt specifications. A preservation analysis confirms this: only 3.05% of temporal expressions survive verbatim in generic summaries, compared to ∼8% for named entities and ∼5% for event mentions.

A one-sentence prompt modification recovers +0.314 [0.254, 0.375] judge accuracy on category-2 (temporal) questions (matched set, direction confirmed in all ten conversations), without materially reducing multi-hop or single-hop performance. The temporal protection is a targeted fix, not a general compression relaxation.

Temporal anchors should be treated as a protected category in any gist compression component. Category-level evaluation paired with preservation rate analysis provides a more complete picture of what a memory strategy preserves — and what it leaves behind.

## Acknowledgements

The authors used Claude (Anthropic) for assistance with literature review, drafting, and experimental design.

## References

P. Chhikara, D. Khant, S. Aryan, T. Singh, and D. Yadav. Mem0: Building production-ready AI agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

C. Cruz. Adaptive focus memory for language models. arXiv preprint arXiv:2511.12712, 2025.

J. Fang, X. Deng, H. Xu, Z. Jiang, Y. Tang, Z. Xu, S. Deng, Y. Yao, M. Wang, S. Qiao, H. Chen, and N. Zhang. LightMem: Lightweight and efficient memory-augmented generation. arXiv preprint arXiv:2510.18866, 2025.

M. Kang et al. ACON: Optimizing context compression for long-horizon LLM agents. arXiv preprint arXiv:2510.00615, 2025.

N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157– 173, 2024.

A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, and Y. Fang. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pages 13851–13870, 2024.

J. L. McClelland, B. L. McNaughton, and R. C. O’Reilly. Why there are complementary learning systems in the hippocampus and neocortex. Psychological Review, 102(3):419–457, 1995.

D. Patel and S. Patel. ENGRAM: Effective, lightweight memory orchestration for conversational agents. arXiv preprint arXiv:2511.12960, 2025.

G. Tononi and C. Cirelli. Sleep function and synaptic homeostasis. Sleep Medicine Reviews, 10(1):49–62, 2006.

C. Wang and J. V. Sun. Unable to forget: Proactive interference reveals working memory limits in LLMs beyond context length. arXiv preprint arXiv:2506.08184, 2025.

S. Wang, E. Yu, O. Love, T. Zhang, T. Wong, S. Scargall, and C. Fan. MemMachine: A ground-truth-preserving

memory system for personalized AI agents. arXiv preprint arXiv:2604.04853, 2026.

Y. Xie. Learning to forget: Sleep-inspired memory consolidation for resolving proactive interference in large language models. arXiv preprint arXiv:2603.14517, 2026.

## A Worked Gist Abstraction Example

Original session: “I finally went to the LGBTQ support group last night — 7 May. It was really moving. I spoke for about ten minutes about my experience coming out to my parents.”

SWC-Full: “Caroline attended an LGBTQ support group and shared her coming-out experience, finding it emotionally meaningful.”

SWC-Temporal: “On 7 May, Caroline attended an LGBTQ support group and spoke for ten minutes about her coming-out experience, finding it emotionally meaningful.”

SWC-Full loses the date (7 May) and duration (ten minutes). Named entities and the event are preserved in both; only temporal anchors differ.

## B Baseline Results Across All Ten Conversations

Table 5: Overall judge accuracy, all ten conversations (includes category 5; Table 1 uses categories 1–4 only).
<table><tr><td>Conv</td><td>Q</td><td>Trunc</td><td>Sliding</td></tr><tr><td>0</td><td>197</td><td>0.355</td><td>0.386</td></tr><tr><td>1</td><td>102</td><td>0.422</td><td>0.461</td></tr><tr><td>2</td><td>191</td><td>0.298</td><td>0.330</td></tr><tr><td>3</td><td>237</td><td>0.304</td><td>0.354</td></tr><tr><td>4</td><td>240</td><td>0.317</td><td>0.329</td></tr><tr><td>5</td><td>158</td><td>0.367</td><td>0.367</td></tr><tr><td>6</td><td>189</td><td>0.249</td><td>0.344</td></tr><tr><td>7</td><td>233</td><td>0.283</td><td>0.343</td></tr><tr><td>8</td><td>192</td><td>0.297</td><td>0.328</td></tr><tr><td>9</td><td>196</td><td>0.260</td><td>0.316</td></tr><tr><td>Pooled</td><td>1935</td><td>0.305</td><td>0.347</td></tr></table>

Sliding window matches or exceeds truncation across all ten conversations (conversation 5 is a tie at 0.367).

## C SWC-Reactive Results

SWC-Reactive (token-threshold triggering only, API default temperature, conversations 0–2) achieved aggregate

judge accuracy of 0.452 vs SWC-Full’s 0.462. Scheduling differences are not a validated contribution of this paper.

## D Category 5 (Adversarial) Results

Category 5 questions are designed to be unanswerable; correct responses contain “no information available.” Conditions removing more content score higher (truncation: 0.72–0.86 across conversations). This reflects content retention, not memory quality.

## E Judge Prompt and Validation

Judge prompt (temperature 0):

Question: {question}. Gold answer: {gold\_answer}. Model answer: {model\_answer}. Does the model answer correctly answer the question, given the gold answer? Reply only YES or NO.

Validation: 60-question blind sample; 95% agreement (57/60); 100% on category 2 (17/17); all 3 disagreements conservative.

Category examples: Cat. 1 (multi-hop): “What fields would Caroline likely pursue?” Cat. 2 (temporal): “When did Caroline go to the LGBTQ support group?” Cat. 3 (open-domain): “Would Caroline likely have Dr Seuss books?” Cat. 4 (single-hop): “What did the charity race raise awareness for?” Cat. 5 (adversarial): Unanswerable questions; correct response acknowledges lack of information.