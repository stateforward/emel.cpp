# Phase 237: Direct Tensor Staged Offset Contract Repair - Context

**Gathered:** 2026-05-08
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure repair)

<domain>
## Phase Boundary

Repair the direct `model/tensor::event::request_staged_load` route so nonzero
`file_offset` source-window behavior is source-backed and tested through public
`process_event(...)`. The maintained `model/loader -> io/loader -> io/staged_read`
route already pre-windows the source span; this phase closes the direct tensor route
gap found by `.planning/v1.26-MILESTONE-AUDIT.md`.

</domain>

<decisions>
## Implementation Decisions

### Locked Audit Findings
- Add a failing public direct tensor staged-load doctest for nonzero `file_offset`
  against a whole-file source buffer before changing implementation.
- Align direct tensor staged-load source-span construction with the maintained
  `io/loader` route unless source inspection proves a stricter pre-windowed contract
  is already enforced everywhere.
- Keep tensor residency ownership in `model/tensor`; `io/staged_read` remains copy-only.
- Preserve explicit `_done` and `_error` publication through public dispatch.
- Model artifact, benchmark, and snapshot updates are approved only when changed source
  requires maintained regeneration.

### Claude's Discretion
- Choose the narrowest repair that satisfies the audit and existing SML rules.
- Add guardrail assertions if useful, but avoid widening staged-read scope into
  file-backed syscalls, async/coroutines, device strategies, or model-family work.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/model/tensor/lifecycle_tests.cpp` already has direct
  `request_staged_load` success/error doctests and helpers for staged-read actor
  injection.
- `src/emel/io/loader/actions.hpp` already applies `source_bytes + tensor.file_offset`
  before dispatching single staged reads.
- `src/emel/io/staged_read/actions.hpp` copies from `source_span` directly and does
  not apply `file_offset` inside single-window copy actions.

### Established Patterns
- State machine behavior is proven through doctest public `process_event(...)`
  calls and `is(...)` state inspection.
- Requirement reopening is already recorded in `.planning/REQUIREMENTS.md` for
  `TNX-01`, `TNX-03`, `TNX-04`, `TST-01`, and `TST-02`.
- Changed-file quality gates must use `EMEL_QUALITY_GATES_CHANGED_FILES` because the
  worktree contains unrelated edits.

### Integration Points
- `src/emel/model/tensor/actions.hpp`
- `src/emel/model/tensor/events.hpp` if event contract comments need clarification.
- `tests/model/tensor/lifecycle_tests.cpp`
- `.planning/v1.26-MILESTONE-AUDIT.md` for final evidence reconciliation.

</code_context>

<specifics>
## Specific Ideas

Use a source like `abcdefgh` or `abcdefghijklmnop`, a nonzero offset such as `2`,
and assert the target receives the offset subspan (`cdef...`) through direct
`model::tensor::event::request_staged_load`.

</specifics>

<deferred>
## Deferred Ideas

`ESG-02B` file-backed open/seek/read taxonomy remains deferred until a future
approved file-backed staged-read source path owns handle/syscall behavior.

</deferred>
