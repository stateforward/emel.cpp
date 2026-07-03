# Phase 230: Context Cleanness and Per-Attempt Lifetime - Context

**Gathered:** 2026-05-07  
**Status:** Ready for closeout

<domain>
## Phase boundary

Phase 230 proves three actor invariants for `emel::io::staged_read` without widening staged-read scope:

- **STG-07:** `staged_read::context` carries no dispatch-local request mirrors.
- **LIFE-02:** Per-attempt request/source/target/callback/status data remains same-RTC stack/event-only.
- **SNR-01:** Staged-read remains copy-only and never claims tensor residency ownership.

This phase is evidence and invariant hardening only; no new staged file-descriptor ownership path is introduced.

</domain>

<decisions>
## Implementation decisions

- Keep `action::context` as an empty persistent actor context.
- Keep per-attempt carriers scoped to `sm::process_event(...)` stack (`status` + runtime wrapper).
- Prove behavior through public `process_event(...)` doctests and source scans; avoid `detail.hpp` reach-through tests.
- Keep done payload semantics as caller-owned buffer echo (`target_buffer`, `bytes_committed`) with no residency claims.

</decisions>

<code_context>
## Existing code insights

- `src/emel/io/staged_read/context.hpp` already defines `struct context {};`.
- `src/emel/io/staged_read/sm.hpp` creates per-attempt `status` and runtime wrapper on stack in `process_event`.
- `src/emel/io/staged_read/events.hpp` keeps request payload and callbacks on the request/event surface.
- `tests/io/staged_read/lifecycle_tests.cpp` already uses callback-observed outcomes through public dispatch.

</code_context>

<specifics>
## Specific ideas

- Add static/public-surface assertions for empty context and callback payload identity.
- Add explicit public tests showing done payload reports caller-owned target pointer and committed bytes only.

</specifics>

<deferred>
## Deferred ideas

- Staged OS-handle acquisition/release proofs remain future-phase work because current staged-read design uses caller-provided `source_span` bytes and owns no kernel handle.

</deferred>
