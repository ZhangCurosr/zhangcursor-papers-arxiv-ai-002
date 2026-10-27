# Silent and Costly: Characterizing and Detecting Prompt-Caching Mis-Integration in LLM Agents

Ying Zhang<sup>⋆</sup> Wake Forest University Winston-Salem, USA ying.zhang@wfu.edu

Yuekang Li University of New South Wales Sydney, Australia yuekang.li@unsw.edu.au

Ruoqi Guo<sup>⋆</sup>   
Griffith University   
Gold Coast, Australia   
rokie.g77@griffith.edu.au

Yi Liu<sup>§</sup> Griffith University Gold Coast, Australia yi.liu@griffith.edu.au

Yepang Liu Southern University of Science and Technology Shenzhen, China liuyp1@sustech.edu.cn

Gelei Deng Nanyang Technological University Singapore gelei.deng@ntu.edu.sg

Leo Zhang   
Griffith University   
Gold Coast, Australia   
leo.zhang@griffith.edu.au

Abstract—Large language model (LLM) services process massive volumes of user requests, resulting in substantial inference costs. To reduce both cost and latency, providers have introduced prompt caching. However, correctly integrating prompt caching is non-trivial due to heterogeneous provider-specific APIs and continuously evolving caching semantics. To understand prompt-caching mis-integration in downstream applications, we conduct the first empirical study of 120 prompt-caching-related pull requests from 10 popular open-source agent systems. Our findings show that prompt-caching mis-integrations arise in both request-side misconfiguration and read-side usage miscount. More importantly, most mis-integrations fail silently without causing observable runtime failures.

Motivated by these findings, we design CACHELINT, which models a prompt-caching mis-integration as an invocation site together with the preconditions required for correct cache integration. CACHELINT locates caching invocation sites with CodeQL and discharges each precondition either against a dated providercapability oracle or with an LLM guard check. CACHELINT achieves 0.71 precision and 0.78 recall in identifying the 120 labeled mis-integrations across the 10 studied agent systems. Our evaluation further demonstrates the effectiveness of CACHELINT in real-world settings, identifying 32 mis-integrations across 571 agent projects with 0.86 precision. These findings have received 21 developer confirmations and led to 18 upstream fixes, including one in Anthropic’s official SDK.

Index Terms—Large Language Models (LLMs), Prompt Caching, Static Analysis, LLM Agents.

## I. INTRODUCTION

Large language model (LLM) services process billions of requests from end users every day, incurring considerable inference costs [1]. Prior work [2] shows that a substantial fraction of prompts are identical, such as system prompts and few-shot examples used in multi-turn conversations. When such prompts are decomposed into multiple requests to LLM services, the same prefix is repeatedly sent across requests. Consequently, LLMs repeatedly reprocess these identical prompt portions, leading to redundant computation and substantial token costs. By mid-2025, enterprise-level LLM API costs were estimated to have reached \$8.4 billion [3].

To address this inefficiency, LLM providers have introduced prompt caching [4]–[6], which reduces redundant computation over identical prompt prefixes by reusing previously computed intermediate results. Specifically, user prompts are cached based on their token prefixes; a cache hit occurs when the incoming request shares an exact token-prefix match with a previously cached prompt, starting from the first token (i.e., byte-for-byte equality). In this case, the cached computation is reused instead of recomputing the prefix. Otherwise, a cache miss occurs, and the prompt is processed normally while its prefix may be inserted into the cache for future reuse. Any modification, insertion, or reordering invalidates the match from that point onward [4], [7].

However, correctly integrating prompt caching is non-trivial because provider-specific configuration settings vary and the underlying mechanisms evolve. For example, Anthropic [4] and Amazon Bedrock [8] require applications to explicitly opt into caching and enforce limits on the number of cache entries per request, whereas OpenAI and Gemini apply caching automatically [5], [6]. Different providers adopt different caching enablement settings. Moreover, providers’ prompt caching semantics can evolve over time. For instance, Bedrock introduced an opt-in one-hour cache lifetime tied to specific models [9], while Anthropic later reduced the default cache duration from one hour to five minutes, which significantly affected hit rates in unchanged deployments [10].

Existing studies have investigated prompt defects and prompt caching performance, but have overlooked the integration issues of prompt caching from the application side. For instance, several existing studies [11]–[13] investigate software defects that degrade the performance and reliability of LLM-integrated systems, including inference engines or agent orchestration frameworks. They do not consider prompt caching a distinct source of integration defects. Later, Tian et al. [14] identified prompt caching as a literature-derived subtype of performance and efficiency defects in a prompt defect taxonomy, while Lumer et al. [15] evaluated prompt caching strategies for reducing latency and API cost. Neither of these studies investigates prompt cache misses caused by application-level integration.

![](images/4697b8927ed03304ee95831c7230ee3320cef9b5d7c944824f7cbfaad037469b.jpg)  
Fig. 1. The overview of our work.

In this paper, we conduct the first empirical study of promptcaching mis-integration in open-source agent frameworks, analyzing application code that violates provider requirements for enabling, configuring, or accounting for prompt caching. As described in Fig. 1, we identify 581 repositories through GitHub search and select the 10 most popular open-source agent applications for analysis. We then collect 148,947 merged pull requests (PRs) from these repositories and analyze their prompt-caching-related code. After inspection, we obtain 120 such defects. Our study explores the following research questions (RQs):

• RQ1 (Defect Type and Root Cause): What are the common types of prompt caching defects in open-source agents, and what are their root causes?

• RQ2 (Defect Symptom): What are the common symptoms of prompt caching integration defects in opensource agents?

• RQ3 (Defect Fix): How do agent developers fix prompt caching integration defects in practice, and what are the common patterns?

Our study uncovers three key findings. First, prompt caching integration defects are concentrated on the request side. Of the 120 defects, 97 corrupt outgoing requests, while the remaining 23 preserve requests but misparse cache-usage metrics. Interestingly, we find that the heterogeneous provider-specific caching mechanisms account for 47 request-side defects, while inconsistent application-side specification of prompt prefixes (e.g., system prompts) is another major cause of cache misses. Second, prompt-caching mis-integration is often silent. In 108 mis-integration cases, defects produce no errors because the model recomputes outputs without using cached tokens. The resulting cache misses manifest only as increased cost and latency in usage metrics. Third, the mis-integration fixes are typically small and localized to request-construction code. These findings suggest that prompt-caching mis-integrations often preserve functional outputs and evade dynamic test oracles, making static analysis a well-suited detection approach.

In this paper, we develop CACHELINT, a static analysis technique for detecting prompt-caching mis-integrations in agent applications. As shown in Fig. 1, CACHELINT pairs each prompt-caching invocation site with its required preconditions, derived from the fixing patterns of our empirical study. Given an agent application codebase, CACHELINT uses CodeQL [16] to identify prompt-caching API invocation sites and extract their surrounding code context. It then checks each associated precondition by either comparing provider-capability preconditions against a dated capability oracle or prompting an LLM (gpt-5.4-mini) with the invocation-site context to assess prefix-stability and accounting preconditions. When a precondition check fails, CACHELINT reports the corresponding invocation site and suggests a fix.

We evaluate CACHELINT on the 10 studied projects, using 120 fixed defects as ground truth. CACHELINT reports 132 warnings, of which 94 are true positives, achieving a precision of 0.71 and a recall of 0.78. To assess its generalizability, we further evaluate CACHELINT on 571 real-world agent repositories, where it achieves a precision of 0.86 and uncovers 32 previously unknown prompt-caching mis-integrations. At the time of reporting, maintainers had confirmed 21 reports and fixed 18 defects. These results demonstrate the practical value of our empirical findings and CACHELINT.

In summary, we make the following contributions:

• We present the first empirical study of prompt-caching mis-integrations, characterizing their causes, symptoms, and fixes. We release our dataset to support future research [17].

• We propose a prompt-caching integration model that captures common mis-integration patterns and generalizes developer practices for automated detection and repair.

• We develop CACHELINT, a static analysis-based detector for identifying prompt-caching mis-integrations. CACHELINT achieves 0.71 precision and 0.78 recall on the 120 labeled mis-integrations. Furthermore, on 571 real-world agent applications, CACHELINT uncovers 32 previously unknown defects at 0.86 precision, resulting in 21 maintainer-confirmed reports and 18 fixes.

## II. BACKGROUND

## A. Prompt Caching

Prompt caching is an optimization technique that reduces LLM inference cost and latency by reusing the computation of previously processed prompt prefixes. Specifically, when a new prompt begins with a prefix that exactly matches a cached prefix, the provider reuses the cached computation instead of recomputing the prefix and charges cached tokens at a substantially reduced rate [4]–[6].

Prompt caching is effective for LLM agents because of frequent prompt reuse. In each call, the agent constructs a prompt from a stable preamble, typically a system instruction, tool definitions, and retrieved or environment context, followed by the conversation history [18]. Across consecutive calls, these prompts share a long common prefix, while only the latest user message or model response changes. Without prompt caching, the agent repeatedly transmits and the LLM repeatedly processes thousands of identical prompt tokens. By reusing these stable prefixes, prompt caching can reduce cached input costs to roughly one-tenth of the standard price and decrease inference latency by up to 80% [1], [4]–[6]. Consequently, modern agent frameworks are increasingly designed to maximize cache reuse [19].

## B. Prompt Caching Integration

Prompt caching integration refers to how applications construct and configure prompts to satisfy provider-defined requirements for enabling caching behavior. When correctly integrated, prompt caching enables reuse of previously computed prompt prefixes, resulting in cache hits; otherwise, the system falls back to full recomputation.

The correct integration of prompt caching places two requirements on applications. First, the application must maintain prefix stability. As providers perform byte-level matching from the first token, the prompt prefix must exactly match a previously cached prefix to guarantee cache hits. Even minor modifications to the prefix can invalidate cache reuse from that point onward [4], [5], [7]. Second, the application must correctly configure provider-exposed caching parameters. For example, settings such as cache prefix length and cache lifetime determine whether a request is eligible for caching and whether cached entries can be reused. Insufficient prefix length or short cache windows may prevent reuse of previously computed results, leading to cache misses even when prompts are structurally identical.

![](images/72986df6c2692950f2bfb88856a26b955eec0c0b9e700f27dd0bd60209ed32dc.jpg)  
Fig. 2. A prompt-caching mis-integration in zylon-ai/private-gpt: a sub-second timestamp at the front of the system prompt breaks the cached prefix on every call; the fix coarsens it to day granularity.

## III. MOTIVATING EXAMPLE

We introduce a real prompt caching integration issue from PrivateGPT [20]. PrivateGPT is an open-source retrievalaugmented generation (RAG) framework [21] that enables users to interact with private documents using an LLM through a context-aware question-answering interface.

The developer attempted to enable the agent to answer queries about the current time while benefiting from prompt caching. The original implementation prepends datetime.now().isoformat() to the system prompt. Since cache hits require an exact match of the prompt prefix from the first token, the changing timestamp invalidates cache reuse for all subsequent stable content (left side of Fig. 2). As a result, despite the provider correctly implementing prompt caching, the application fails to reuse cached prompt prefixes and repeatedly recomputes the stable guidelines and retrieved context. Additionally, because the agent continues to produce correct outputs without raising any errors, this mis-integration remains silent until increased inference cost or a reduced cache-hit rate is observed.

To address this issue, the developer coarsens the timestamp to date granularity using .date(), which enables the prompt prefix to remain stable throughout the day while still providing accurate time information (right side of Fig. 2).

We define such prompt-caching mis-integration as application code that violates provider requirements for enabling, configuring, or accounting for prompt caching. Such defects can prevent cache reuse, cause incorrect cache accounting, or even trigger runtime failures. Despite their practical impact, real-world reports show that many of these defects remain unnoticed in production until operational metrics, such as cache-hit rates or API costs, are examined [7], [22]. This gap motivates our empirical study and the development of automated detection of prompt-caching mis-integration.

Listing 1. Model-name gating, the largest cause (cline#3343,   
simplified): caching keyed on a model name; the fix keys on a   
provider capability.   
1. block = msg.setdefault("content", {})   
2. if model.id.startswith("claude") or "anthropic" in   
model.id:   
3. + if registry.supports\_cache(model.id):   
4. block["cache\_control"] = {"type": "ephemeral"}

TABLE I  
TOP TEN AGENT APPLICATIONS (#PC = MERGED PRS THAT FIX A PROMPT-CACHING DEFECT).
<table><tr><td>Project</td><td>#Star</td><td>#PR</td><td>#PC</td></tr><tr><td>OpenClaw [23]</td><td>379,764</td><td>53,078</td><td>39</td></tr><tr><td>Hermes-Agent [24]</td><td>198,679</td><td>37,707</td><td>21</td></tr><tr><td>AutoGPT [25]</td><td>185,051</td><td>8,235</td><td>6</td></tr><tr><td>OpenCode [26]</td><td>176,850</td><td>13,884</td><td>9</td></tr><tr><td>LobeHub [27]</td><td>78,924</td><td>8,479</td><td>1</td></tr><tr><td>OpenHands [28]</td><td>77,888</td><td>10,430</td><td>11</td></tr><tr><td>Cline [29]</td><td>63,600</td><td>5,897</td><td>19</td></tr><tr><td>Goose [30]</td><td>49,956</td><td>7,248</td><td>7</td></tr><tr><td>Aider [31]</td><td>46,526</td><td>1,047</td><td>1</td></tr><tr><td>Nanobot [32]</td><td>44,499</td><td>2,942</td><td>6</td></tr><tr><td>Total</td><td></td><td>148,947</td><td>120</td></tr></table>

## IV. EMPIRICAL STUDY METHODOLOGY

This section presents the data collection and analysis methods we use to answer RQ1–RQ3.

## A. Dataset Collection

To study RQ1–RQ3, we automatically collect real-world agent projects from GitHub, including their source code, issues, and pull requests (PRs) related to prompt-caching defects. We first adopt a keyword-based search strategy using three categories of prompt-caching implementation indicators: provider cache markers (cache\_control, cachePoint, cachedContent, prompt\_cache\_key), usage fields (cached\_tokens, cache\_read, cache\_creation), and the literal phrase “prompt caching”. The search returned 581 repositories.

We then filter out irrelevant repositories and retain candidates that meet three selection criteria: (1) scope— generalpurpose conversational or coding agent applications for end users, excluding inference engines, editors with an incidental agent, and single-purpose utilities; (2) active maintenance— non-archived repositories with commits within three months of our collection snapshot (2026-05-10); (3) prompt-caching integration—verified source-level prompt-caching implementations, excluding unrelated caches (e.g., HTTP, session, and key-value caches). These criteria yield 581 candidate repositories.

To bound the manual annotation effort, we rank the remaining repositories by GitHub stars and keep the top ten. These criteria yield the ten subjects in Table I. They span TypeScript, Python, and Rust and cover the major providers: Anthropic, OpenAI, Gemini, Bedrock, and OpenRouter-style gateways.

Identifying prompt-caching mis-integration. Within these subjects, we identify each defect through the merged PR that fixes it. Defects without a merged fixing PR provide no root cause or fix strategy to analyze and are therefore excluded. We collect 148,947 merged PRs across the ten subjects and query the GitHub Search API using prompt-caching-related keywords, restricted to type:pr is:merged created:<=2026-05-10. This search returned 622 candidate merged PRs.

Because keyword matching is noisy, three authors manually screened all 622 candidates to separate genuine defects from false matches, reading each candidate’s title, body, and, where needed, its diff. The first three authors independently screened all candidate PRs, achieving a Fleiss’ κ of 0.76. Disagreements were resolved through discussion. Screening retained 120 relevant prompt-caching PRs and discarded 502 false matches involving unrelated caches, model additions, dependency updates, documentation, and generic token accounting. An independent second pass over all candidates identified no additional false matches among the retained PRs. Table I summarizes the dataset used in our study.

## B. Data Analysis and Annotation

To understand prompt-caching integration defects and answer our research questions, we proceed in two steps. First, for each defect we (1) collect the fixing code, (2) recover the link between the defect report and the fixing code, and (3) collect and study the discussion thread. Second, three authors, all agent-application developers with at least two years of agent development and usage experience, apply open coding [33], an established qualitative research method, to classify each defect by its root cause, symptom, and fixing strategy. Each author coded all defects independently, and disagreements were resolved by consensus.

## V. EMPIRICAL STUDY RESULTS

## A. RQ1: Defect Type and Root Causes

Table II shows that 97 defects (81%) are on the request side, and the remaining 23 (19%) are on the read side.

1) Request Side: Request-side mis-integration occurs when an agent constructs or configures a request incorrectly, causing the provider to recompute content that could otherwise be served from the cache, thereby incurring the full inference cost, or causing the request to fail at runtime.

Provider compatibility. The agent adds a provider-specific caching configuration that is unsupported by the selected model or provider, causing the provider to ignore the configuration and leave prompt caching inactive. Two root causes account for this failure.

First, model-name gating enables caching based on hardcoded model identifiers rather than provider capabilities. As a result, a model rename or the addition of a new model can silently disable caching. Listing 1 shows a model-name-based mis-integration in cline#3343. Specifically, an upstream rename changed a Gemini 2.5 model ID, but the application enabled caching only for models

TABLE II  
TAXONOMY OF THE 120 PROMPT-CACHING DEFECTS: TWO TYPES AND EIGHT ROOT CAUSES, EACH WITH ITS COUNT AND SHARE, MECHANISM, MOSTCOMMON SYMPTOM, AND MOST COMMON FIX. BOLD ROWS GIVE GROUP SUBTOTALS.
<table><tr><td>Type</td><td>Root cause</td><td></td><td></td><td># % Description</td><td>Symptom</td><td>Fixing strategy</td></tr><tr><td></td><td>Provider compatibility</td><td></td><td>47 39</td><td></td><td></td><td></td></tr><tr><td></td><td>Model-name gating</td><td></td><td></td><td>37 31 A hardcoded model-name check guards caching; a new model fails it.</td><td>Silent</td><td>Capability lookup</td></tr><tr><td>Request side</td><td>Incompatible marker</td><td></td><td></td><td>10 8 The cache marker&#x27;s format is one the provider rejects.</td><td>Crash</td><td>Marker guard</td></tr><tr><td>97 (81%)</td><td>Prefix stability</td><td>41 34</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Prefix reassembly</td><td></td><td></td><td>15 13 Each call rebuilds the prefix in a different order, breaking the byte-exact match.</td><td>Silent</td><td>Cache-boundary fix</td></tr><tr><td></td><td>Volatile value</td><td></td><td></td><td>13 11 A per-request value (timestamp, UUID) is embedded in the cached prefix.</td><td>Silent</td><td>Cache-boundary fix</td></tr><tr><td></td><td>Nondeterministic serialization 10</td><td></td><td>8</td><td>Identical content serializes to different bytes (e.g.,</td><td>Silent</td><td>Prefix canonicalization</td></tr><tr><td></td><td>Misplaced marker</td><td></td><td></td><td>unsorted map keys). 3 3 The cache marker is placed on a changing message, not the stable prefix.</td><td>Silent</td><td>Marker guard</td></tr><tr><td></td><td>Cross-turn mutation</td><td></td><td></td><td>9 8 Program state mutated between turns invalidates a prefix an earlier turn cached.</td><td>Silent</td><td>Lifecycle guard</td></tr><tr><td>Read side 23 (19%)</td><td>Usage miscount (read-side)</td><td></td><td>23 19</td><td>Caching works, but returned cache-token counts are summed or shown incorrectly.</td><td>Wrong report</td><td>Usage-counter fix</td></tr></table>

Listing 2. Volatile value in the cached prefix (goose#2434, simplified): a per-turn timestamp in the system prompt; the fix moves it out of the cached span.

1. system\_prompt = [   
2. f"Current time: {now.isoformat()}",   
3. "You are a coding agent. ...",   
4. ]   
5. + # per-turn time now rides the (uncached) user   
message   
6. + messages.append({"role": "user",   
7. + "content": f"[time] {now.isoformat()}"})

whose names began with claude or contained anthropic. As a result, the renamed model no longer passed the check, and caching was disabled. Similar failures occurred in OpenHands#8365, where a -latest tag was removed; hermes-agent#17425, which excluded Mini-Max’s cacheable models; and OpenHands#4704, where an overly restrictive condition disabled caching altogether. Second, an incompatible marker occurs when the agent attaches a provider-specific caching marker that the selected provider or model does not support. For example, nanobot#2687 attached Anthropic cache\_control markers to requests for every model accessed through OpenRouter. Because GPT, Gemini, and DeepSeek models do not support this marker, the requests resulted in 401 errors and a runtime crash.

Prefix stability. The agent’s own prefix is not stable from one call to the next, so it never matches the prefix the previous call cached. Four root causes destabilize the prefix. A volatile value embeds a per-request datum in the cached span: as shown in Listing 2, goose#2434 placed a live timestamp in the cached system prompt, changing the prefix on every call so the cache never hit. Similarly, nanobot#1115 did the same with a minute-level clock, and in each case the fix moves the volatile datum out of the cached span. Prefix reassembly rebuilds the prefix in a changed order or shape each call, as in hermes-agent#754, where a gateway reassembled the system prompt every turn and lifted cost by roughly 75%. Nondeterministic serialization emits different bytes for identical content, as in openclaw#64198’s per-instance map order and hermes-agent#3991’s random uuid4 call-id. A misplaced marker sits on a changing message rather than on the stable prefix.

Cross-turn mutation. A cross-turn mutation lets a stateful change between turns rewrite a fragment an earlier turn had already cached, so the next call writes the cache again instead of reading it. In openclaw#58038, pruning history images after three turns rewrote the already-cached conversation prefix mid-session, so each compaction silently rebuilt the cache instead of hitting it.

2) Read Side: Read-side mis-integrations occur when caching works exactly as intended, but the agent mishandles the cache usage the provider reports back in the response.

Usage miscount. The provider reports cache usage in its response, but the agent incorrectly parses, aggregates, prices, or displays the returned cache-token counts. In cline#10266, the cache-creation tokens were hardcoded to zero in two provider handlers, so the usage readout reported no caching even as the cache kept hitting.

## Answer to RQ1

We observed eight root causes of prompt-caching misintegrations. Provider compatibility and prefix stability account for the majority. Integrating prompt caching correctly across various providers is challenging.

## B. RQ2: How Defects Manifest

Table II shows that 108 of the 120 defects (90%) remain silent. They raise no exception, no failed assertion, and no crash, and the model output stays correct while only caching degrades. We discuss the symptoms below.

Within these no-error defects, the vast majority, 86 (72%), are silent waste: the cache stops hitting and the run continues unchanged, so the only trace is extra spend on a hosted API or added latency on the application, visible only on a usage meter. cline#3343 is typical, where a renamed model slipped past a caching gate and the bill climbed with no other sign. The remaining 22 (18%) of the no-error defects produce wrong telemetry: caching works as intended, but the cachetoken counts reported to the developer are parsed, summed, or displayed incorrectly, so the meter a developer would trust to confirm caching is itself wrong.

Only 12 of the 120 defects (10%) crash, and they concentrate in incompatible markers (8 of the 12); the other four are cross-turn mutations (2), a misplaced marker (1), and a usage miscount (1). A crash is the one symptom an ordinary oracle catches, because it surfaces as an explicit error the first time the request runs: OpenHands#3689 hit Vertex’s Extra inputs not permitted rejection on a cache\_control field, hermes-agent#1216 drew a 400 from a cache marker on an empty text block, and AutoGPT#12789 raised a TypeError on a null cachetoken field. A single run or test exposes each, whereas the other 90% give no such signal.

In summary, detecting prompt-caching mis-integration is hard precisely because 90% of the defects fail silently: cline#3343 let a renamed model slip past a caching gate while the bill simply climbed, goose#2434 placed a live timestamp in the system prompt so the cache never hit, and cline#10266 hardcoded cache-creation tokens to zero so even the usage meter under-reported caching. None of them raises an exception or changes the model’s answer, so crashand assertion-based oracles miss the majority by construction.

## Answer to RQ2

Prompt-caching mis-integrations are overwhelmingly silent: 108 of 120 (90%) raise no error and surface only as extra spend or added latency, which makes oracles hard to define for testing this class of defect.

## C. RQ3: Fixing Strategies

We studied the 120 merged fixes together with the discussion in their linked reports to explore (1) the common patterns and (2) the complexity of the fixing patches.

Common fixing patterns. As shown in Fig. 3, the eight root causes map to six recurring fixing strategies. Each strategy involves a relatively small, localized repair. Capability lookup resolves caching from a provider-capability check instead of a hardcoded model name; the marker guard attaches the cache marker only in the shape and on the message the provider accepts; a cache-boundary fix moves changing content out of the cached span (or rebuilds that span in a fixed order) so the prefix stays byte-identical; prefix canonicalization serializes the prefix deterministically, e.g., by sorting map keys, so identical content always produces identical bytes; a lifecycle guard stops a later turn from rewriting a prefix an earlier turn already cached; and a usage-counter fix corrects the arithmetic that parses, sums, or displays the returned cache-token counts.

![](images/5536bc264b088015d99b72d8daf184d4a3d3c3ab845c73e92fb3c48613b44135.jpg)  
Fig. 3. Root cause vs. fixing strategy for the 120 defects. Each cell is a count, shaded by group and darker for larger counts.

Fixing strategies are largely predictable from the underlying root causes. Simply assigning each root cause its most common fixing strategy correctly predicts 95 of the 120 fixes (79%), with U(fix | rc) = 0.72. Five causes map to one strategy every time: usage miscounts to usage-counter fixes (23/23), volatile value to cache-boundary fixes (13/13), serialization to prefix canonicalization (10/10), cross-turn mutation to lifecycle guards (9/9), and misplaced markers to a marker guard (3/3). Only three cross-cutting causes spread their repairs: model-name gating splits between a capability lookup (21 of 37) and a marker guard (13), with a lifecycle guard (2) and prefix canonicalization (1) for the rest; an incompatible marker fans across four strategies, mostly a marker guard (6) with prefix canonicalization (2) and one each to a capability lookup and a lifecycle guard; and prefix reassembly divides among cache-boundary fixes (10), lifecycle guards (3), and prefix canonicalization (2). These three account for all 25 modal-fix errors, so the fix is largely, though not perfectly, a relabeling of the cause.

Patch complexity. Although the strategy varies by cause, the patches themselves are small. Four of the six strategies are static, local, or report-only: capability lookups (22, 18%), marker guards (22, 18%), prefix canonicalization (15, 13%), and report-only usage-counter fixes (23, 19%), and together they cover 82 of the 120 fixes (68%); only cache-boundary fixes (23, 19%) and lifecycle guards (15, 13%) reach into runtime structure. The most common fix simply adds or corrects a single guard or the arithmetic on already-correct token counts.

Diagnosing a prompt-caching defect is the hard part: because it is mostly silent, the work lies in noticing it at all and tracing a cost or latency regression back to the request-building code. The fix itself is usually simple and patterned: most (68%) are small, static, local edits that recur across projects as reusable templates.

## VI. METHODOLOGY

Based on our empirical findings, we develop CACHELINT, an automated technique for detecting prompt-caching misintegrations in agent codebases. CACHELINT models both prompt-caching-related APIs and the integration contexts associated with each class of mis-integration. Given an agent codebase, it performs static analysis to identify and report prompt-caching mis-integrations. We first present the underlying integration model and then describe the detection workflow.

## A. Prompt-Caching Integration Modeling

The 120 mis-integration cases arise from diverse root causes, but they share a common pattern: each either writes or reads the prompt cache incorrectly across providers. This regularity admits a uniform formalization. We model each pattern of prompt-caching mis-integration as a pair of a promptcaching invocation site m and a caching-context precondition j, written $P = \langle m , j \rangle$ , where:

• m is a program point that sends content to, or reads an accounting field from, the cache. It is either a requestside site that attaches a cache directive to a prefix span (a marker from K = {cache\_control, cachePoint, cachedContent, a ttl/cacheRetention setting, prompt\_cache\_key}) or a readside site that consumes a usage field from U {cache\_read, cache\_creation, cached\_tokens, cacheReadInputTokens}.

• j is a conjunction of three condition families, one per root-cause group in Table II and generated by the grammar below: prefix stability (STAB), provider capability (CAP), and accounting safety (ACCT). The other two causes, cross-turn mutation and prefix reassembly, fall outside the grammar: each compares prefixes across different calls or turns, which a single program state cannot check.

⟨j⟩ ::= ⟨stab⟩ | ⟨cap⟩ | ⟨acct⟩ | ⟨j⟩ and ⟨j⟩   
⟨stab⟩ ::= volatile\_value\_reaches(span) // clock/uuid/rng   
| nondet\_serialization(span) // unsorted keys   
| marker\_on\_unstable(span) // changing span   
⟨cap⟩ ::= predicate\_gates\_marker(model\_lit) and   
dated\_oracle(model,prov,ttl,t) ̸= predicate   
| marker\_shape\_rejected(marker,prov,t)   
⟨acct⟩ ::= usage\_consumed\_without\_guard(U) // int+None

We define a mis-integration M as an invocation reached on a path where a required caching precondition is not guaranteed to hold. Formally, $M = \langle m , j , \lnot \mathrm { G U A R D } \rangle$ : an invocation site m, an unchecked conjunct $j \in \{ \mathrm { S T A B , C A P , A C C T } \}$ , and no preceding guard that discharges j before control reaches m. A guard discharges j if it checks the relevant condition, normalizes the inputs so that j holds, or blocks the invocation when j is false. Among the three conjuncts, CAP is unique because its truth depends on a time-varying external condition rather than the program itself. It may change long after the code is released without any modification to the implementation. For example, when Bedrock introduced a one-hour TTL, previously correct code silently became defective, whereas in other cases unchanged code became correct.

Algorithm 1 CACHELINT: detecting a prompt-caching mis  
integration at a cache-commit site.   
Require: source A; implemented pairs P; dated oracle O   
Ensure: warnings W   
1: $W  \emptyset$   
2: for all $\langle m , j \rangle \in P$ do   
3: for all $s \in \operatorname { L o c A T E } ( A , m )$ do ▷ CodeQL / regex   
4: $\mathbf { i f } \ j = \mathbf { C A P }$ then   
5: v ← ORACLEDIFF(ASSUMPTION(s), O) ▷ dated,   
non-LLM   
6: else ▷ STAB or ACCT   
7: h ← LLMGUARDCHECK(s, CONTEXT(A, s), j)   
8: $v  h \mathit { \Delta } ?$ OK : BUG   
9: end if   
10: if v ̸= OK then   
11: ${ \dot { W } }  W \cup \{ \langle s , j , v ,$ FIXHINT(m, j)⟩}   
12: end if   
13: end for   
14: end for   
15: return W

Consider cline#3343 (Listing 1), which illustrates the model. At invocation site m, the code attaches cache\_control only when model.id.startswith("claude") holds. On provider prov, gemini-2.5 is cacheable, but its identifier fails this gate, so cache\_control is dropped and the prefix is recomputed at full price. In the model, this mis-integration is the pair ⟨m, CAP⟩ whose conjunct dated\_oracle(gemini-2.5, prov, \_, $\qquad \mathtt { t } ) \neq$ predicate is violated: the dated oracle marks gemini-2.5 cacheable while the predicate admits only identifiers that begin with claude. Mined from fixed defects, the same pair flags other sites where the predicate excludes a model the oracle marks cacheable.

## B. Context-Guided Mis-Integration Identification

Not every pair in Section VI-A is worth detecting. Each pair becomes a static rule, and a rule earns its place only when it fires reliably and recurs across projects, so CACHE-LINT implements the anti-patterns that are (i) locatable, in that the invocation site m is a single, statically inspectable program point, and (ii) recurrent, in that the pair generalizes across providers and apps rather than turning on one project’s internals. This is the mechanizable subset of the taxonomy (Section ${ \mathrm { V I I } } { \mathrm { I } } { \mathrm { I } } - { \mathrm { A } } ) { \vdots }$ we leave out the multi-snapshot causes, crossturn mutation and prefix reassembly, whose context a single program state cannot express.

![](images/94e35b2efa382fed071b66613c3f44aeb77e13289e23549d6b630b567a4d14fa.jpg)  
Fig. 4. CACHELINT’s findings on 33 real-world projects, placed by code size (KLOC) and popularity (stars).

Algorithm 1 takes an application’s source, the implemented pairs, and the dated capability oracle O, and emits a warning for each unguarded invocation. For each pair it locates the invocation sites of the marker or meter, and for each site it discharges the pair’s conjunct in one of two ways. A CAP conjunct is settled off the program: CACHELINT extracts the code’s capability assumption ⟨site,provider scope, capability,predicate⟩ and diffs it against O, a dated external lookup the model never performs. A STAB or ACCT conjunct is settled on the program: CACHELINT extracts the invocation site’s surrounding context and asks an LLM whether a discharging barrier reaches m, such as a datetruncation or sort\_keys canonicalizer for STAB or an or 0 or accumulator guard for ACCT. This LLM guard check replaces the inter-procedural backward slice a static detector would use, which is brittle under the dynamic dispatch and reflection common in agent code. When no barrier discharges the conjunct, CACHELINT emits the site, the verdict, and a suggested patch class drawn from RQ3’s cause-to-fix map (a capability lookup for model-name gating, a cache-boundary fix for a volatile value, a usage-counter fix for a usage miscount), so every warning ships with the modal repair developers already apply.

For each CAP assumption, CACHELINT diffs the code’s predicate against a provider-capability matrix carrying an explicit last\_verified date and a sourced behavior for each provider on an unsupported directive (reject, ignore, or unknown). When the code treats a model as capable but the matrix does not (an over-claim), the verdict is a hard BUG if the provider rejects the directive, SUSPECTED if its behavior is unknown, and benign if it silently ignores it; when the matrix marks the model as capable but the code excludes it (an under-claim), the verdict is a silent LATENT miss. Because the matrix is dated and external, the same one-hour-TTL code shape resolves to a BUG on Bedrock and an OK on the Anthropic API, a distinction no static rule makes (Section VII-B). Provider changes therefore reach CACHELINT through two places: this matrix, and the marker list K and usage-field list U that the CodeQL queries behind LOCATE match. Every matrix entry is sourced from the provider’s official documentation and stamped with the date of that check; when a provider changes its caching behavior, we update the affected entries and re-run CACHELINT, leaving the CodeQL queries and the guard-check prompts unchanged.

## C. Implementation

The LOCATE step is a set of CodeQL queries [16], one per anti-pattern and language (Python, JavaScript/TypeScript, Ruby, Go, C#). The guard check is delegated to a frontier LLM, gpt-5.4-mini, prompted with the invocation site and its enclosing function and asked only whether a discharging barrier reaches m. As with prior static detectors [34], this check is not sound: the LLM can overlook a real guard or posit a spurious one, so CACHELINT may raise false warnings or miss defects. We release our source code at our website [17].

## VII. EVALUATION

In this section, we evaluate CACHELINT on real-world agent applications and investigate the following research questions:

• RQ4 (Detection Effectiveness). How effective is CACHELINT at detecting prompt-caching misintegrations?

• RQ5 (Ablation Study). How does each component of CACHELINT contribute to its overall effectiveness?

• RQ6 (Real-world Application). Can CACHELINT provide useful information to help developers identify prompt-caching mis-integrations?

Datasets. We evaluate CACHELINT on two datasets, both drawn from the 581 repositories collected in Section IV-A.

Dataset A consists of the 10 agent applications from which we derived our detection model, together with the 120 misintegrations labeled in them. Every defect in Dataset A is linked to a merged fixing PR, so the dataset provides ground truth and supports measuring both precision and recall. Because the detection rules were mined from these same fixes, the recall measured on Dataset A is an in-sample upper bound rather than an independent estimate. Dataset B consists of the remaining 571 repositories. It is disjoint from Dataset A, and therefore held out from rule derivation, and it is also broader, since it retains the agent SDKs, frameworks, and single-purpose tools that the empirical study excluded. No exhaustive defect labels exist for these repositories. We apply CACHELINT to the latest version of every repository in both datasets and manually inspect each warning in its source context to label it a true or false positive.

Evaluation metrics. We label a warning a true positive (TP) when the flagged site genuinely violates a prompt-caching precondition and a false positive (FP) otherwise; on Dataset A, a labeled defect that CACHELINT misses is a false negative (FN). We report precision = TP/(TP + FP) on both datasets, and recall = TP/(TP + FN) on Dataset A only, since Dataset B gives no denominator of known defects.

All experiments were conducted on a MacBook Pro equipped with an Apple M5 Max processor and 48 GB of RAM. We present the experimental results below.

## A. RQ4: Detection Effectiveness

Effectiveness on Dataset A. On Dataset A, CACHELINT reports 132 potential mis-integrations, comprising 94 TP and 38 FP, and fails to report 26 of the 120 labeled defects. It thus achieves a precision of 0.71 and a recall of 0.78.

We manually inspected the code associated with the missed defects. We found that most false negatives stem from defects that require reasoning across multiple invocation sites or turns, which are beyond the scope of the single-site model in CACHELINT. Of the 120 labeled defects, 96 occur at a single deterministic program point and are amenable to static inspection. These include model-name gating, usage miscounts, volatile cached-prefix values, nondeterministic serialization, misplaced markers, and incompatible markers. The remaining 24 defects require comparing prompts across calls or turns. They account for 24 of the 26 false negatives (15 involving prefix reassembly and 9 involving cross-turn mutation). The other two false negatives are in-scope volatile-value defects in which the changing values are introduced outside the enclosing function and thus fall beyond the context inspected by the guard check. Within its stated scope, CACHELINT recovers 94 of the 96 in-scope defects, demonstrating near-complete coverage of defects amenable to single-site analysis.

Effectiveness on Dataset B. CACHELINT reports 37 potential mis-integrations across the 571 repositories of Dataset B; Fig. 4 shows 32 TP and 5 FP across 33 projects, a precision of 0.86. The detected defects cover a wide range of agent repositories, from flagship frameworks such as n8n (194,717

GitHub stars) to single-maintainer tools, with codebases ranging from 4 to 4,297 KLOC. Most findings fall into two categories: model-name gating and volatile values in the cached prefix. The remaining cases involve marker-budget and usagemiscount defects.

Two examples illustrate the results. In dirac, prompt caching was gated on a hard-coded Claude model in a switch statement, the same pattern as in our motivating cline example. The maintainers later merged a capabilitybased fix. In n8n, a millisecond timestamp was prepended to the cached system prompt, causing the prefix to change on every request and preventing cache hits. These results show that CACHELINT can detect prompt-caching mis-integrations in both large and widely used agent projects.

The five false positives match the syntactic pattern of a mis-integration but are benign. Two occur because the caching requirement has already been satisfied: in litellm, cache-token accounting is safely guarded with .get(..., 0), and in unsloth, the system prompt uses a daylevel date().isoformat(), which is stable enough for caching. In gemini-cli, a Date.now() value is used only as a request identifier and never appears in the cached prompt prefix. The remaining two arise from conservative matching: tldraw uses a provider feature flag that resembles model-name gating, and pydantic-ai contains a hardcoded model name only in test code. These cases are outside the scope of the current static rules and could be eliminated with more precise path-sensitive analysis.

We reproduced all 32 true positives. For the 5 crash defects, we triggered the runtime error. For the 27 silent defects, we inspected request/response pairs and the returned cache-usage fields (e.g., cache\_read and cached\_tokens).

## Answer to RQ4

CACHELINT achieves a precision of 0.71 and a recall of 0.78 on the 10 popular agent applications, and a precision of 0.86 on 571 real-world agent applications. The anti-patterns mined from fixed prompt-caching defects help CACHELINT detect unknown mis-integrations in real-world agent projects, from flagship frameworks to single-maintainer tools.

## B. RQ5: Ablation Study

CACHELINT uses two procedures to discharge caching preconditions in Algorithm 1. A CAP conjunct is checked against the dated capability oracle, while STAB and ACCT conjuncts are checked by the LLM-based guard check. To measure the contribution of each procedure, we construct two ablated variants and run them on Dataset B. Each ablated variant removes one component and falls back to gpt-5.4-mini without the context that the component provides. CACHELINT w/o guard check judges every located STAB or ACCT site with a generic API-misuse prompt instead of the targeted barrier question, for example whether a date() truncation or a sort\_keys canonicalizer reaches the site. CACHELINT w/o oracle judges every located CAP site from the LLM’s own knowledge. Table III reports the results.

The guard check is necessary for filtering benign STAB and ACCT sites. Without it, precision drops from 0.86 to 0.31 because the detector reports many candidates whose caching preconditions have already been satisfied. For example, unsloth includes a timestamp in the system prompt, but the timestamp is coarsened to day granularity with date().isoformat(), making it stable enough for caching. The full CACHELINT recognizes this guard and does not report the site, whereas CACHELINT w/o guard check flags it as a volatile-value defect. Similarly, litellm safely accounts for cache tokens by guarding the lookup with .get(..., 0) before summing the values; the ablated variant reports this benign site.

The dated capability oracle is also necessary. Without it, the model-name-gating query reports 349 candidate sites across the scanned repositories, most of which are benign branches in routers, registries, and tests. Precision drops to 0.05. The oracle distinguishes an actual capability mismatch from a harmless model gate. In dirac, for example, the oracle marks a renamed model as cacheable, while the code predicate excludes it; this disagreement exposes a real defect. In contrast, when a gated-out model is truly not cacheable, the code agrees with the oracle and the site is cleared. The oracle is also needed because these facts are time-sensitive: when asked about the 33 dated one-hour-TTL capability facts encoded in the matrix, gpt-5.4-mini answers only five confidently and correctly, and must abstain on models released after its training cutoff.

Both ablations also lower the number of true positives, from 32 to 18 without the guard check and to 17 without the oracle (Table III). Without the component’s context, the LLM tends to treat a prompt-caching mis-integration as an ordinary API call, so it misses real defects while still flagging many benign sites, which lowers both true positives and precision. The full CACHELINT is stable across runs: over five runs, precision deviated between 0.0 and +0.02, and the deviation came from a single intermittently reported false positive.

## Answer to RQ5

Both the LLM-based guard check and the dated capability oracle are necessary for CACHELINT’s effectiveness. The guard check filters benign STAB and ACCT sites, while the oracle filters benign CAP sites. Removing either component sharply reduces precision.

## C. RQ6: Real-World Application

To evaluate the practical value of CACHELINT, we reported the 32 true-positive findings from RQ4 to project maintainers and tracked their responses. At the time of writing, 21 reports have been confirmed and 18 have been fixed; 11 are awaiting a response and none has been rejected. The confirmations include widely used projects such as AutoGPT, n8n, and Zed.

TABLE III ABLATION OF CACHELINT.
<table><tr><td>Configuration</td><td>TP</td><td>FP</td><td>Precision</td></tr><tr><td>CACHELINT (full)</td><td>32</td><td>5</td><td>0.86</td></tr><tr><td>w/o guard check</td><td>18</td><td>41</td><td>0.31</td></tr><tr><td>w/o oracle</td><td>17</td><td>349</td><td>0.05</td></tr></table>

The confirmed cases span provider SDKs, large frameworks, and single-maintainer tools. We discuss three representative examples.

Usage miscount in the Anthropic SDK. anthropic-sdk-csharp is Anthropic’s official .NET SDK. The API returns cache-creation and cache-read token counts as optional usage fields, but the SDK deserialized these fields into nullable properties and consumed them without a null check. When a response omits the fields, clients cannot reliably account for cache usage or cache savings. We reported the issue, and the maintainers fixed it.

Model gating in altimate-code. altimate-code attached cache\_control markers only for Anthropic-native calls, using a hard-coded set of model paths. Identical requests routed through OpenRouter- or Bedrock-style gateways were therefore left unmarked and could not use prompt caching. This is the same request-side gating pattern as in the motivating cline case. We reported the issue, and the maintainers fixed it by replacing the model-list check with a capabilitybased check.

Volatile prefix in iTerm2. iTerm2 is a widely used macOS terminal. Its assistant integration placed a request-varying value near the beginning of the cached system prompt. This changed the cached prefix on every call and prevented cache hits. We reported the issue, and the maintainer fixed it by moving the volatile value out of the cached region.

## Answer to RQ6

CACHELINT can provide useful information to help developers diagnose and fix prompt-caching misintegrations.

## VIII. THREATS TO VALIDITY

Construct and internal validity. Our results depend on how defects are identified, labeled, and validated. To reduce noise from keyword search, three authors independently screened all candidate PRs and identified 120 genuine prompt-caching defects. Three experienced raters also independently coded all defects and resolved disagreements by consensus. To reduce evaluation bias, we inspected each CACHELINT warning in source context and reported findings upstream, where maintainers confirmed 21 reports and fixed 18. LLM judgments and provider capabilities may change over time. We mitigate this by fixing the prompts and model version, using a dated capability snapshot, and evaluating on 571 repositories. Recall on the 571 repositories is not measured because the total number of prompt-caching defects there is unknown; the 0.78 recall on the labeled corpus is an in-sample proxy, since the same corpus informed the detection rules.

External validity. Our taxonomy is derived from the ten most-starred multi-provider agent applications on GitHub and may not generalize to closed-source or non-GitHub projects. We mitigate this by evaluating CACHELINT on 571 disjoint real-world repositories. CACHELINT covers 96 (80%) of the defects in our dataset; multi-snapshot defects remain out of scope. The empirical study was restricted to end-user agent applications and excluded SDKs and frameworks, which may bias the taxonomy toward application code; however, the 571-repository evaluation set includes SDKs, and the confirmed defect in Anthropic’s official .NET SDK (Section VII-C) shows that the same patterns occur in provider-maintained code. CACHELINT targets misintegrations that involve explicit cache markers or parameters, including OpenAI’s prompt\_cache\_key and Gemini’s cachedContent. When a provider enables caching automatically on the model side, request-side defects are observable only through the response, for example as consistently zero cached tokens; such implicit caching is outside CACHELINT’s current scope and is left for future work. As provider caching semantics evolve, our results reflect the 2026- 05-10 capability snapshot. The taxonomy is separated from provider-specific API details; as caching APIs evolve, the oracle entries and detection rules are updated from providers published documentation, as described in Section VI-B.

## IX. RELATED WORK

Compatibility, fragmentation, and API misuse. Providerspecific caching behavior is related to platform compatibility and API-misuse research. FicFinder detects Android compatibility issues through API-context pairs [34], [35], while CiD studies API evolution [36]. Prompt caching introduces an additional dimension: application-side request invariants such as prefix stability and cache-marker accounting. Like API-misuse detectors [37], CACHELINT checks whether program behavior satisfies an external API contract. Rather than relying on usage patterns alone, it combines data-flow analysis with provider knowledge. Prompt-caching mis-integration also resembles a configuration error. Studies of misconfiguration show that such errors can cause production failures and often arise from unchecked assumptions about the deployment environment or software version [38]–[41]. Model-name gates and cachemarker shapes are assumptions of this kind, written into code rather than into configuration files, and CACHELINT checks them against a dated external oracle.

Empirical bug studies and silent defects. Repositorymining studies have characterized faults in deep-learning software [42]–[44], LLM inference engines [11], and agent frameworks [12], [13]. We follow the same open-coding methodology [33] to analyze 120 prompt-caching fixes. Prompt-caching defects belong to the broader class of silent performance bugs: they preserve functional correctness while increasing cost or latency, escaping traditional testing and crash-based oracles [45]–[47].

LLM-assisted program analysis. Recent analyses combine LLM reasoning with static verification. InferROI infers resource intentions and validates them with control-flow analysis [48]; IRIS infers taint specifications and grounds them with CodeQL [49]. CACHELINT adopts the same paradigm for prompt caching: CodeQL identifies candidate invocation sites [16], an LLM guard check verifies prefix stability and usage accounting, and a dated capability oracle determines whether provider-specific caching assumptions remain valid.

Prompt caching and LLM serving. Prior systems optimize prompt caching inside the serving engine through KV-cache management [50], cross-request prefix reuse [51], and modular prompt reuse [2]. We instead study failures introduced by application code, where unstable prefixes or unsupported cache markers violate the engine’s caching preconditions before a request reaches the provider. Closer to our topic, Tian et al. [14] list prompt caching as one category in a taxonomy of prompt defects, and Lumer et al. [15] measure how caching strategies trade off cost against latency on long-horizon agentic tasks. Both treat caching as a defect category to catalog or as a strategy to tune; we instead characterize the defects that arise when applications integrate caching incorrectly and build a detector that flags them in code.

## X. CONCLUSION

In this paper, we presented the first empirical study of prompt-caching mis-integrations in LLM agent applications. We investigated 120 real defects from 10 open-source agent systems to characterize their root causes, symptoms, and fixes. We found that these defects arise on both the request and read sides, mostly fail silently while preserving model output, and are typically repaired by small, localized edits. Guided by these findings, we proposed a prompt-caching integration model that pairs each caching invocation site with the preconditions for its correct use, and built CACHELINT, a static analysis technique that discharges each precondition against a dated provider-capability oracle or an LLM guard check. On 571 real-world repositories, CACHELINT detected 32 previously unknown mis-integrations at 0.86 precision, yielding 21 developer confirmations and 18 upstream fixes. In the future, we plan to extend CACHELINT’s oracle to more providers and to automatically generate and validate fixes.

## ACKNOWLEDGMENT

We thank the anonymous reviewers for their constructive feedback and guidance, which helped us improve this paper.

## DATA AVAILABILITY

We release the dataset and artifacts used in this work at our website [17].

## REFERENCES

[1] ProjectDiscovery and P. Malhotra, “How we cut LLM costs by 59% with prompt caching,” ProjectDiscovery engineering blog, 2026, https: //projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching.

[2] I. Gim, G. Chen, S.-s. Lee, N. Sarda, A. Khandelwal, and L. Zhong, “Prompt cache: Modular attention reuse for low-latency inference,” in Proceedings of Machine Learning and Systems (MLSys), 2024.

[3] Menlo Ventures, “2025 mid-year LLM market update: Foundation model landscape and economics,” Menlo Ventures Perspective, 2025, enterprise LLM-API spend reached \$8.4B in 1H 2025, more than doubling in six months. https://menlovc.com/perspective/2025-mid-year-llm-market-u pdate/.

[4] Anthropic, “Prompt caching,” Anthropic documentation, https://docs.a nthropic.com/en/docs/build-with-claude/prompt-caching.

[5] OpenAI, “Prompt caching,” OpenAI documentation, https://platform.o penai.com/docs/guides/prompt-caching.

[6] Google, “Context caching,” Google Gemini API documentation, https: //ai.google.dev/gemini-api/docs/caching.

[7] M. Brenndoerfer, “Caching for LLMs: Prompt, semantic, and invalidation,” Technical blog, 2026, https://mbrenndoerfer.com/writing/cachin g-prompt-semantic-invalidation-hit-rates-llm.

[8] Amazon Web Services, “Prompt caching for faster model inference,” Amazon Bedrock documentation, https://docs.aws.amazon.com/bedroc k/latest/userguide/prompt-caching.html.

[9] ——, “Amazon Bedrock now supports 1-hour duration for prompt caching,” AWS What’s New announcement, 2026, one-hour cache TTL, gated to specific models (Claude Sonnet/Haiku/Opus 4.5). https://aws. amazon.com/about-aws/whats-new/2026/01/amazon-bedrock-one-hou r-duration-prompt-caching/.

[10] A. Whoff, “Anthropic silently dropped prompt cache TTL from 1 hour to 5 minutes,” DEV Community, 2026, https://dev.to/whoffagents/anthro pic-silently-dropped-prompt-cache-ttl-from-1-hour-to-5-minutes-16ao.

[11] M. Liu et al., “A first look at bugs in LLM inference engines,” arXiv preprint arXiv:2506.09713, 2025.

[12] Z. Xue, Y. Zhao, S. Wang, K. Chen, and H. Wang, “A characterization study of bugs in llm agent workflow orchestration frameworks,” in 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE), 2025, pp. 3369–3380.

[13] X. Zhu et al., “An empirical study of bugs in modern LLM agent frameworks,” arXiv preprint arXiv:2602.21806, 2026.

[14] H. Tian, C. Wang, B. Yang, L. Zhang, and Y. Liu, “A taxonomy of prompt defects in LLM systems,” arXiv preprint arXiv:2509.14404, 2025, https://arxiv.org/abs/2509.14404.

[15] E. Lumer, F. Nizar, A. Jangiti, K. Frank, A. Gulati, M. Phadate, and V. K. Subbiah, “Don’t break the cache: An evaluation of prompt caching for long-horizon agentic tasks,” arXiv preprint arXiv:2601.06007, 2026, https://arxiv.org/abs/2601.06007.

[16] GitHub, “CodeQL,” https://codeql.github.com, 2026.

[17] CacheLint Authors, “CacheLint: Dataset and artifacts,” Project website, 2026, https://sites.google.com/view/prompt-caching-mis-integration.

[18] MindStudio, “What is prompt caching in Claude code? how to save millions of tokens,” Technical blog, 2026, https://www.mindstudio.ai/bl og/prompt-caching-claude-code-token-savings.

[19] esengine, “DeepSeek-Reasonix: A DeepSeek-native AI coding agent for your terminal, engineered around prefix-cache stability,” GitHub repository, 2026, https://github.com/esengine/DeepSeek-Reasonix.

[20] Zylon by PrivateGPT, “PrivateGPT,” May 2023. [Online]. Available: https://github.com/zylon-ai/private-gpt

[21] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel, S. Riedel, and D. Kiela,¨ “Retrieval-augmented generation for knowledge-intensive nlp tasks,” in Proceedings ofthe 34th International Conference on Neural Information Processing Systems, ser. NIPS ’20. Red Hook, NY, USA: Curran Associates Inc., 2020.

[22] 0xAlcibiades, “[bug]: Honcho auto-injected context rebuilds cached system prompt every n turns, invalidating kv cache on every prefixcaching backend,” Hermes-Agent, GitHub issue #13631, 2026, https: //github.com/NousResearch/hermes-agent/issues/13631.

[23] “OpenClaw,” GitHub repository, https://github.com/openclaw/openclaw.

[24] “Hermes-Agent,” GitHub repository, https://github.com/NousResearch/ hermes-agent.

[25] “AutoGPT,” GitHub repository, https://github.com/Significant-Gravitas/ AutoGPT.

[26] “OpenCode,” GitHub repository, https://github.com/anomalyco/opencod e.

[27] “LobeHub,” GitHub repository, https://github.com/lobehub/lobehub.

[28] “OpenHands,” GitHub repository, https://github.com/OpenHands/Ope nHands.

[29] “Cline,” GitHub repository, https://github.com/cline/cline.

[30] “Goose,” GitHub repository, https://github.com/aaif-goose/goose.

[31] “Aider,” GitHub repository, https://github.com/Aider-AI/aider.

[32] “Nanobot,” GitHub repository, https://github.com/HKUDS/nanobot.

[33] C. B. Seaman, “Qualitative methods in empirical studies of software engineering,” IEEE Transactions on Software Engineering, vol. 25, no. 4, pp. 557–572, 1999.

[34] L. Wei, Y. Liu, and S.-C. Cheung, “Taming Android fragmentation: Characterizing and detecting compatibility issues for Android apps,” in Proc. 31st IEEE/ACM Int. Conf. Automated Software Engineering (ASE), 2016, pp. 226–237.

[35] L. Wei, Y. Liu, S.-C. Cheung, H. Huang, X. Lu, and X. Liu, “Understanding and detecting fragmentation-induced compatibility issues for Android apps,” IEEE Transactions on Software Engineering, vol. 46, no. 11, pp. 1176–1199, 2020.

[36] L. Li, T. F. Bissyande, H. Wang, and J. Klein, “CiD: Automating the´ detection of API-related compatibility issues in Android apps,” in Proc. 27th ACM SIGSOFT Int. Symp. Software Testing and Analysis (ISSTA), 2018, pp. 153–163.

[37] S. Amann, H. A. Nguyen, S. Nadi, T. N. Nguyen, and M. Mezini, “A systematic evaluation of static API-misuse detectors,” IEEE Transactions on Software Engineering, vol. 45, no. 12, pp. 1170–1188, 2019.

[38] Z. Yin, X. Ma, J. Zheng, Y. Zhou, L. N. Bairavasundaram, and S. Pasupathy, “An empirical study on configuration errors in commercial and open source systems,” in Proceedings of the 23rd ACM Symposium on Operating Systems Principles (SOSP), 2011, pp. 159–172.

[39] T. Xu, J. Zhang, P. Huang, J. Zheng, T. Sheng, D. Yuan, Y. Zhou, and S. Pasupathy, “Do not blame users for misconfigurations,” in Proceedings of the 24th ACM Symposium on Operating Systems Principles (SOSP), 2013, pp. 244–259.

[40] T. Xu and Y. Zhou, “Systems approaches to tackling configuration errors: A survey,” ACM Computing Surveys, vol. 47, no. 4, pp. 70:1–70:41, 2015.

[41] S. Zhang and M. D. Ernst, “Which configuration option should I change?” in Proceedings of the 36th International Conference on Software Engineering (ICSE), 2014, pp. 152–163.

[42] Y. Zhang, Y. Chen, S.-C. Cheung, Y. Xiong, and L. Zhang, “An empirical study on TensorFlow program bugs,” in Proceedings of the 27th ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA), 2018, pp. 129–140.

[43] N. Humbatova, G. Jahangirova, G. Bavota, V. Riccio, A. Stocco, and P. Tonella, “Taxonomy of real faults in deep learning systems,” in Proceedings of the 42nd International Conference on Software Engineering (ICSE), 2020, pp. 1110–1121.

[44] M. J. Islam, G. Nguyen, R. Pan, and H. Rajan, “A comprehensive study on deep learning bug characteristics,” in Proceedings of the 27th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE), 2019, pp. 510–520.

[45] G. Jin, L. Song, X. Shi, J. Scherpelz, and S. Lu, “Understanding and detecting real-world performance bugs,” in Proceedings of the 33rd ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), 2012, pp. 77–88.

[46] M. Selakovic and M. Pradel, “Performance issues and optimizations in JavaScript: An empirical study,” in Proceedings of the 38th International Conference on Software Engineering (ICSE), 2016, pp. 61–72.

[47] F. Tambon, A. Nikanjam, L. An, F. Khomh, and G. Antoniol, “Silent bugs in deep learning frameworks: An empirical study of Keras and TensorFlow,” Empirical Software Engineering, vol. 29, no. 1, p. 10, 2024.

[48] C. Wang, J. Liu, X. Peng, Y. Liu, and Y. Lou, “Boosting static resource leak detection via LLM-based resource-oriented intention inference,” in Proc. 47th IEEE/ACM Int. Conf. Software Engineering (ICSE), 2025, arXiv:2311.04448.

[49] Z. Li, S. Dutta, and M. Naik, “IRIS: LLM-assisted static analysis for detecting security vulnerabilities,” in International Conference on Learning Representations (ICLR), 2025.

[50] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for

[51] L. Zheng et al., “SGLang: Efficient execution of structured language model programs,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.

large language model serving with PagedAttention,” in Proceedings of the 29th Symposium on Operating Systems Principles (SOSP), 2023.