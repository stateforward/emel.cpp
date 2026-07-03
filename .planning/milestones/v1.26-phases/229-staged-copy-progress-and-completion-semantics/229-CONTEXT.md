# Phase 229: Staged Copy Progress and Completion Semantics - Context

**Gathered:** 2026-05-07  
**Status:** Ready for verification-led closeout

<domain>
## Phase boundary

Phase 229 owns actor-level staged copy semantics for `emel::io::staged_read`:

- **STG-04:** deterministic per-stage contiguous byte copy into caller target region.
- **STG-05:** full logical-span completion with strictly monotone forward progress.
- **STG-06:** exactly one terminal success outcome for a complete successful run.

All behavior selection remains in `guards.hpp` + `sm.hpp`; copy execution remains bounded in
`actions.hpp` with no queue/defer/coroutine mechanics and no dispatch-time heap allocation.

</domain>

<decisions>
## Implementation decisions

- Keep `action::context` empty; do not mirror dispatch-local request payload in context.
- Treat caller-provided `source_span` as authoritative input bytes for this phase.
- Require exact `source_span_bytes == logical_byte_length` before copy acceptance.
- Publish success via one `events::staged_window_done` callback carrying
  `bytes_committed == logical_byte_length`.

</decisions>

<code_context>
## Existing code insights

- `src/emel/io/staged_read/sm.hpp` already wires explicit validation-decision states and a
  success row that lands in `effect_publish_staged_window_done`.
- `src/emel/io/staged_read/actions.hpp` performs deterministic chunk tiling:
  full-size segments first, then optional tail segment, all through `std::memcpy`.
- `tests/io/staged_read/lifecycle_tests.cpp` already includes:
  - exact-division staged copy proof,
  - non-divisible remainder proof,
  - rejection of invalid `source_span` contract,
  - single-dispatch completion checks through public `process_event`.

</code_context>

<canonical_refs>
## Canonical references

- `.planning/ROADMAP.md` (Phase 229 goal/success criteria)
- `.planning/REQUIREMENTS.md` (STG-04/STG-05/STG-06)
- `AGENTS.md`
- `docs/rules/sml.rules.md`
- `src/emel/io/staged_read/{sm,guards,actions,events}.hpp`
- `tests/io/staged_read/lifecycle_tests.cpp`

</canonical_refs>

<deferred>
## Deferred ideas

- Context lifetime and handle ownership constraints remain in Phase 230 (STG-07/LIFE-02/SNR-01).
- Deterministic error taxonomy expansion remains in Phase 231.

</deferred>

---
*Phase: 229-staged-copy-progress-and-completion-semantics*
