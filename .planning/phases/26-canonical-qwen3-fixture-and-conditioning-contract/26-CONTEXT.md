# Phase 26: Canonical Qwen3 Fixture And Conditioning Contract - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 26 must lock one official Qwen3-0.6B GGUF truth anchor and one explicit maintained
request-conditioning contract before the runtime cutover happens. The maintained paritychecker and
benchmark surfaces must stop silently inheriting the old Llama fixture/prompt assumptions, but
this phase still stays narrow to the canonical `Qwen3-0.6B-Q8_0.gguf` slice and one explicit
formatter contract derived from that model's primary chat template.

Phase 26 does not require successful Qwen generation or a successful maintained benchmark compare
run. Until runtime support lands in Phase 27, the maintained paritychecker and benchmark paths
must fail explicitly on the canonical Qwen fixture and contract instead of silently reusing the
old Llama slice or pretending Qwen generation already works.

This phase stays inside `tests/models/`, the maintained tool surfaces in `tools/paritychecker` and
`tools/bench`, the existing injected formatter seam used by generator and conditioner, and the
planning artifacts that define the operator-facing contract. User approval has now been given to
widen the formatter-facing request shape from one flat prompt string to structured chat messages so
the maintained path can represent the primary GGUF chat template honestly. This phase still must
not claim named-template selection, broader Qwen-family support, or generic template behavior
beyond the canonical maintained slice.

</domain>

<decisions>
## Implementation Decisions

### Fixture Truth Anchor
- Treat `tests/models/Qwen3-0.6B-Q8_0.gguf` as the only maintained v1.6 generation fixture, with
  explicit checksum, source, and download provenance in `tests/models/README.md`.
- Update maintained tool-facing fixture identity and help text so drift fails on the old Llama
  anchor instead of silently reusing it.
- Keep the maintained acceptance boundary on the existing paritychecker and benchmark surfaces
  rather than inventing a Qwen-only harness.
- Before runtime support exists, require maintained paritychecker and benchmark setup to fail
  explicitly on the canonical Qwen fixture instead of drifting back to the old Llama success path.

### Formatter Input Contract
- Widen the maintained formatter-facing request shape now from one flat prompt string to explicit
  structured chat messages so the primary GGUF chat template can be represented honestly.
- Keep the existing injected formatter seam as the integration boundary for generator and
  conditioner, but update the request payload model as needed to carry structured messages.
- Scope the widening narrowly to the maintained generator, paritychecker, and benchmark surfaces
  required for the canonical Qwen3 slice; do not turn this phase into a broad new product surface.

### Template Source And Selection
- Treat GGUF `tokenizer.chat_template` metadata as the formatter source of truth when it is
  present, rather than hard-coding model-specific prompt text in shared formatter code.
- On maintained paths, honor only the primary GGUF `tokenizer.chat_template`; named template
  variants stay deferred.
- Build formatter behavior behind the existing injected `formatter_ctx + format_fn` seam already
  used by generator and conditioner, rather than adding formatter specialization to the shared
  formatter layer.

### Unsupported Template Behavior
- Classify maintained support explicitly: `no_template`, `supported_contract`, or
  `unsupported_template`. Do not silently fall back to raw formatting when a template is declared.
- Hard-fail maintained paritychecker and benchmark setup when the model declares a primary template
  that the current maintained formatter contract does not support.
- Do not provide a maintained-path raw/manual override for unsupported templates.

### Contract Visibility
- Keep the resolved formatter contract versioned and explicit on maintained proof artifacts so old
  baseline or prompt assumptions fail clearly.
- Before runtime support exists, publish the resolved formatter contract in maintained setup or
  failure output so the canonical conditioning assumptions are still operator-visible even when
  generation aborts with a truthful pre-runtime error.

### the agent's Discretion
- The exact split between tool-local metadata extraction, template validation, and injected
  formatter-adapter ownership can stay local as long as the maintained contract is explicit and
  identical on both EMEL and reference paths.
- The exact structured message payload fields can stay additive as long as they are sufficient to
  represent the primary template honestly and do not introduce implicit behavior.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/text/formatter/format.hpp` already provides the maintained `format_fn` seam used by
  generator and conditioner.
- `src/emel/generator/sm.hpp`, `src/emel/generator/context.hpp`, and
  `src/emel/text/conditioner/actions.hpp` already inject and call `formatter_ctx + format_fn`
  without requiring state-machine rewrites.
- `src/emel/generator/events.hpp` and `src/emel/text/conditioner/events.hpp` currently define the
  flat prompt request model that must now be widened with explicit approval.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp` already own the
  maintained generation fixture constants, baseline contract strings, and reference tokenization
  paths.
- `src/emel/model/data.hpp` already has metadata storage for `tokenizer.chat_template` and named
  template variants.
- `tools/paritychecker/paritychecker_tests.cpp` already covers maintained generation-contract
  behavior that can be widened without new test infrastructure.

### Established Patterns
- Maintained generation proof uses explicit fixture identity and contract strings, not ambient
  discovery.
- Prompt formatting is injected through the existing formatter callback seam, so metadata-derived
  adapter binding can stay additive without state-machine rewrites.
- Prior milestone work already prefers explicit published contracts and explicit unsupported-path
  failure over silent degradation or hidden fallback behavior.
- The repo already treats richer Jinja/template behavior as separate text machinery, not something
  hidden inside generation call sites.

### Integration Points
- `tests/models/README.md` is the canonical fixture provenance ledger.
- `src/emel/text/formatter/format.hpp` is the shared callback ABI for EMEL prompt conditioning.
- `src/emel/generator/events.hpp` and `src/emel/text/conditioner/events.hpp` are the maintained
  request-shape boundaries that currently need widening.
- `tools/paritychecker/parity_runner.cpp`, `tools/paritychecker/parity_main.cpp`,
  `tools/paritychecker/paritychecker_tests.cpp`, and `tools/bench/generation_bench.cpp` are the
  maintained operator-facing generation surfaces that currently still encode the Llama/raw anchor.

</code_context>

<specifics>
## Specific Ideas

- The official Qwen3 GGUF metadata uses `general.architecture=qwen3` and ships
  `tokenizer.chat_template`, so the maintained v1.6 contract should derive formatter behavior from
  that metadata rather than from hard-coded model-specific prompt helpers.
- The same injected formatter binding should be used to size EMEL prompt buffers, feed EMEL
  generation, and condition the reference tokenization path.
- The user explicitly approved widening the current flat prompt boundary so maintained formatter
  work can use structured chat messages instead of guessing or silently stripping template
  semantics.
- Named template variants are intentionally deferred; the primary GGUF `tokenizer.chat_template`
  is the only maintained template source for this phase.
- Maintained parity and benchmark output should publish the resolved formatter contract the same way
  prior milestones published explicit runtime contracts.
- The fixture provenance entry should record the downloaded file size `610M` and SHA256
  `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Repo Rules
- `AGENTS.md` — explicit behavior modeling, no implicit fallback, and approved machine-structure
  change boundary
- `docs/rules/sml.rules.md` — RTC actor model and explicit control-flow rules

### Milestone Scope
- `.planning/PROJECT.md` — milestone goal, scope guardrails, and current brownfield constraints
- `.planning/REQUIREMENTS.md` — `FIX-01` and `COND-01`
- `.planning/ROADMAP.md` — Phase 26 boundary and phase ordering

### Current Code Seams
- `src/emel/generator/events.hpp` — current maintained generation request shape
- `src/emel/text/conditioner/events.hpp` — current conditioner request shape
- `src/emel/text/formatter/format.hpp` — injected formatter ABI
- `src/emel/model/data.hpp` — stored GGUF tokenizer template metadata

</canonical_refs>

<deferred>
## Deferred Ideas

- Named-template selection from `chat_template_names` / `chat_template_values`.
- Any maintained raw/manual override path for unsupported templates.
- Full arbitrary `tokenizer.chat_template` rendering via the Jinja parser/formatter stack beyond
  the canonical maintained slice.
- Broader Qwen chat/tool-use product surfaces beyond what the canonical maintained parity/bench
  slice needs.
- Broader Qwen-family support beyond the canonical dense `Qwen3-0.6B-Q8_0.gguf` slice.

</deferred>

---
*Phase: 26-canonical-qwen3-fixture-and-conditioning-contract*
*Context gathered: 2026-03-27*
