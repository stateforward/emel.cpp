# Phase 155: Parity Actor Boundary Enforcement Closure - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Close `LANE-02` by removing paritychecker dependence on actor `actions.hpp`, `guards.hpp`, and
`detail.hpp` internals except for approved kernel-owned arithmetic surfaces.

</domain>

<decisions>
## Implementation Decisions

### Boundary Tightening
- `tools/paritychecker/parity_engines.cpp` currently includes:
  - `emel/gguf/loader/detail.hpp`
  - `emel/kernel/aarch64/actions.hpp`
  - `emel/model/detail.hpp`
  - `emel/model/llama/detail.hpp`
- The generation harness legitimately needs GGUF metadata decoding, model metadata/vocab loading,
  llama execution/audit views, and kernel arithmetic diagnostics.
- Expose those first three needs through public `any.hpp` wrapper surfaces so paritychecker does
  not name loader/model/llama detail namespaces directly.
- Keep AArch64 flash diagnostic comparison behind `emel/kernel/aarch64/detail.hpp` as the approved
  kernel-owned arithmetic surface for this phase.

### Tests
- Expand source checks to scan all paritychecker `.cpp`/`.hpp` files, excluding only
  `paritychecker_tests.cpp`, for actor action/guard/detail includes and namespace reaches.
- The check should allow documented kernel-owned detail surfaces while blocking loader, model,
  generator, tokenizer, formatter, parser, and detokenizer internals.

</decisions>

<code_context>
## Existing Code Insights

- Existing tests only block selected text generator/detokenizer/Jinja internal patterns.
- `parity_engines.cpp` owns the remaining direct reaches called out by the milestone audit.
- `src/emel/model/llama/detail.hpp` already defines the execution/audit structs and helpers used by
  generation diagnostics; a public wrapper can avoid paritychecker naming `detail::`.
- `src/emel/model/detail.hpp` owns GGUF-derived model/vocab loaders used by paritychecker; a public
  wrapper can preserve the implementation without exposing the detail namespace to paritychecker.
- `src/emel/gguf/loader/detail.hpp` owns constants and sizing helpers used by local GGUF metadata
  inspection; a public wrapper can publish the needed constants/sizing API.

</code_context>

<deferred>
## Deferred Ideas

Moving kernel AArch64 flash diagnostic internals out of `actions.hpp` into a narrower kernel detail
implementation can be planned as kernel cleanup later. Phase 155 only requires paritychecker to use
an approved kernel-owned arithmetic surface rather than actor actions directly.

</deferred>
