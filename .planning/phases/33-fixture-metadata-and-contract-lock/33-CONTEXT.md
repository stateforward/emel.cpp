# Phase 33: Fixture, Metadata, And Contract Lock - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 33 locks one exact maintained Liquid fixture, one exact executable metadata truth source, and
one exact maintained request-conditioning contract before any `lfm2` runtime work starts. The phase
stays inside `tests/models/`, maintained generation tool surfaces in `tools/paritychecker` and
`tools/bench`, the existing formatter/conditioner request seam, and the planning artifacts that
define the operator-facing contract.

This phase does not require successful Liquid generation. Until Phases 34 and 35 land explicit
`lfm2` model/runtime support, the maintained paritychecker and benchmark paths must fail explicitly
on the canonical Liquid fixture and contract instead of silently reusing the old Qwen slice,
falling back to `format_raw`, or implying broader Liquid support.

</domain>

<decisions>
## Implementation Decisions

### Fixture Truth Anchor
- **D-01:** Treat `tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf` as the only maintained v1.9
  generation fixture.
- **D-02:** Keep maintained paritychecker and benchmark generation strict on the exact stable path,
  not just filename matching and not arbitrary compatible-looking Liquid fixtures.
- **D-03:** Do not add alternate CLI or env overrides for other Liquid fixtures on the maintained
  path in Phase 33.
- **D-04:** Help and error text should say explicitly that maintained v1.9 generation is locked to
  the exact `Q4_K_M` Liquid fixture for now.

### Metadata Truth
- **D-05:** Treat official GGUF/config metadata as the executable truth source for the maintained
  Liquid slice.
- **D-06:** Record the contradiction that official prose still mentions `32,768` context while
  GGUF/config metadata report `128000`; maintained repo surfaces must follow the executable
  metadata.
- **D-07:** Operator-visible truth should name `lfm2` explicitly rather than speaking in generic
  GGUF or Llama-family terms.

### Maintained Message Envelope
- **D-08:** Lock the maintained Liquid request shape to support `system + user` messages in Phase
  33.
- **D-09:** Keep `system` optional but first-class: when present, preserve it explicitly rather
  than collapsing it into user text.
- **D-10:** Keep maintained proof artifacts single-turn in Phase 33 even though the contract allows
  `system + user`.
- **D-11:** Keep prior `assistant` messages out of the maintained v1.9 contract for now.

### Unsupported-Request Behavior
- **D-12:** Reject named template variants beyond the primary `tokenizer.chat_template` in Phase
  33.
- **D-13:** Hard-fail on tool use or Liquid tool markers on maintained parity/bench paths instead
  of silently stripping them.
- **D-14:** Hard-fail on prior `assistant` messages or any request shape that would require
  `keep_past_thinking` semantics.
- **D-15:** Do not fall back to `format_raw` on maintained Liquid parity/bench surfaces when the
  template is declared but unsupported.

### Operator-Visible Truth Surfaces
- **D-16:** `tests/models/README.md` must record the maintained Liquid fixture provenance and the
  metadata-truth note that `lfm2` / `128000` override stale prose.
- **D-17:** Paritychecker and bench setup/error output must print the resolved Liquid formatter
  contract during Phase 33 even before runtime support exists.
- **D-18:** Setup/error output must also name the exact maintained fixture path, not only the
  contract string.
- **D-19:** Do not expand stored baseline artifact formats in Phase 33; keep baseline-format
  changes deferred until parity and benchmark publication phases.

### the agent's Discretion
- The exact split between README wording, help text, and setup/error publication can stay local as
  long as fixture identity and contract identity are both explicit.
- The exact internal naming of the Liquid formatter binding can stay additive as long as it does
  not blur the maintained boundary with the existing Qwen contract.

</decisions>

<specifics>
## Specific Ideas

- The user wants full LFM support eventually, but that is explicitly not what Phase 33 or v1.9
  means today.
- The user explicitly rescoped v1.9 from `Q8_0` to `Q4_K_M`; this broader quant-runtime claim is
  part of the maintained milestone now.
- The user wants future tool support, but tool-use semantics are deferred out of this phase.
- The shared formatter ABI already carries `system/user/assistant`, but the maintained Liquid slice
  should only bless `system + user` in Phase 33.
- Phase 33 proof artifacts should remain single-turn, but the user wants later proof coverage to
  catch up to the broader contract rather than staying artificially narrow forever.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Repo Rules
- `AGENTS.md` — explicit behavior modeling, no hidden fallback, and approval-sensitive machine
  structure rules
- `docs/rules/sml.rules.md` — RTC actor model and explicit control-flow rules

### Milestone Scope
- `.planning/PROJECT.md` — v1.9 goal, scope guardrails, and maintained Liquid constraints
- `.planning/REQUIREMENTS.md` — `FIX-02`, `META-01`, and `COND-03`
- `.planning/ROADMAP.md` — Phase 33 boundary and success criteria
- `.planning/research/SUMMARY.md` — milestone synthesis and metadata-truth guidance

### Current Code Seams
- `tests/models/README.md` — canonical fixture provenance ledger
- `tools/generation_formatter_contract.hpp` — maintained template/contract binding surface
- `src/emel/text/formatter/format.hpp` — shared formatter callback ABI and message model
- `src/emel/text/conditioner/events.hpp` — conditioner request-shape boundary
- `src/emel/generator/events.hpp` — maintained generation request-shape boundary
- `tools/paritychecker/parity_main.cpp` — maintained generation help text
- `tools/paritychecker/parity_runner.cpp` — maintained fixture identity, formatter-contract
  publication, and explicit unsupported-path behavior
- `tools/bench/generation_bench.cpp` — maintained benchmark fixture identity and formatter-contract
  publication

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/models/README.md`: already records official fixture provenance for maintained Llama and
  Qwen slices and should be extended with the Liquid truth anchor.
- `tools/generation_formatter_contract.hpp`: already has explicit maintained support categories
  (`no_template`, `supported_contract`, `unsupported_template`) and a published contract string.
- `src/emel/text/formatter/format.hpp`: already supports structured `chat_message` input with
  `system/user/assistant` roles.
- `src/emel/text/conditioner/events.hpp` and `src/emel/generator/events.hpp`: already carry
  structured messages plus `add_generation_prompt` and `enable_thinking` flags across the existing
  maintained path.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp`: already resolve
  primary `tokenizer.chat_template` bindings and print formatter-contract metadata on maintained
  setup/output surfaces.

### Established Patterns
- The repo already prefers one official fixture per maintained slice rather than loose model-family
  discovery.
- Prior Qwen work chose explicit unsupported-path failure over silent prompt fallback, and the user
  wants the same strictness for Liquid Phase 33.
- Maintained proof/publication surfaces already expose contract strings as auditable output, so
  Phase 33 should extend that pattern rather than inventing a new visibility mechanism.

### Integration Points
- `tests/models/README.md` is the operator-facing provenance source of truth.
- `tools/paritychecker/parity_main.cpp` and `tools/paritychecker/parity_runner.cpp` are the
  maintained correctness entrypoint and error/help publication surface.
- `tools/bench/generation_bench.cpp` is the maintained benchmark setup/publication surface.
- `tools/generation_formatter_contract.hpp` is the maintained contract matcher and formatter
  adapter seam that Phase 33 will widen from the current Qwen-shaped contract to a Liquid-specific
  one.

### Known Current Concern
- The latest quality-gate run got through `emel_tests` and coverage, then failed in the existing
  paritychecker build because `llama_layer` member names changed upstream in
  `tools/paritychecker/parity_runner.cpp`. This is pre-existing branch debt, not a Phase 33
  decision, but planners should be aware it may affect local verification flow.

</code_context>

<deferred>
## Deferred Ideas

- Full LFM support with no maintained-slice exceptions.
- Liquid tool use / function-calling support.
- Multi-turn history support, including prior `assistant` messages and `keep_past_thinking`
  semantics.
- Proof corpus expansion so later maintained evidence covers more than single-turn prompts.
- A possible GitHub issue or backlog item to track later proof-coverage expansion beyond the Phase
  33 single-turn corpus.
- Any broader fixture override or sibling-quant support.

</deferred>

---
*Phase: 33-fixture-metadata-and-contract-lock*
*Context gathered: 2026-03-31*
