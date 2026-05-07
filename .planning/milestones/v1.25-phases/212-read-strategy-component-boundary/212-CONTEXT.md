---
phase: 212-read-strategy-component-boundary
status: in_progress
requirements:
  - READ-01
created: 2026-05-05T14:18:00Z
last_updated: 2026-05-05T14:18:00Z
---

# Phase 212 Context

Phase 212 establishes `src/emel/io/read` as the canonical Stateforward.SML read/copy
strategy actor under `src/emel/io`. The phase is boundary-only: it must not perform real
file open / seek / read / close calls, must not publish a copied-bytes outcome, must not
model file/offset/length/layout/target-buffer validation logic, and must not take or
share tensor residency ownership. Concrete validation and platform gating, real read
execution + error taxonomy + transient-resource lifetime, tensor integration, public
runtime/evidence exposure, behavior tests + scope guardrails, and milestone publication
are owned by Phases 213-218 respectively.

The structural reference is the v1.24 mmap boundary phase (Phase 204) and the existing
`src/emel/io/mmap` component. Phase 212 mirrors that scaffold for the read/copy
strategy. The read strategy differs semantically from mmap — read writes copied bytes
into a caller-owned target buffer and never publishes a mapped-region descriptor — but
those data-plane specifics are deferred. Phase 212 routes every accepted request to a
fail-closed `unsupported_platform_error_decision` so the boundary actor is observable
through `process_event(...)` without depending on later-phase work.

Locked decisions:

- `src/emel/io/read` is the canonical read/copy-only Stateforward.SML component. Other
  strategies (mmap, staged/chunked constrained-memory, cooperative async, device) remain
  out of scope for v1.25 and out of scope for the read component permanently.
- The component follows the canonical SML layout with component-local `context.hpp`,
  `events.hpp`, `errors.hpp`, `guards.hpp`, `actions.hpp`, `detail.hpp`, and `sm.hpp`
  files and exposes `emel::io::read::sm`. Boundary-only Phase 212 requires no
  `actions.cpp`; concrete read execution + platform code lands in Phase 214 (mirroring
  how `src/emel/io/mmap/actions.cpp` was introduced for Phases 205/206 work, not for
  Phase 204 boundary).
- Tensor residency lifecycle ownership remains with `model/tensor`. The read component
  MUST NOT store dispatch-local request data (request, target-buffer pointer, file
  metadata, phase indices, error codes, status mirrors, output pointers) in
  `read::context`. Per AGENTS.md context rules, dispatch-local data flows through typed
  internal events only.
- `model/loader` remains orchestration-only. Read strategy selection is reached through
  `emel/io` events in Phase 215 / 216, never by loader internals.
- The mmap path is unchanged. No edits to `src/emel/io/mmap/*`. No reroute of
  `emel::io::sm` to read in Phase 212; the additive top-level alias `emel::IoRead`
  exposes the new component without disturbing existing callers.
- Phase 212 does NOT introduce any platform-specific read calls (`read`, `pread`,
  `lseek`, `open`, `close`, `ReadFile`, `CreateFileW`, `std::ifstream`, `fread`,
  `fopen`, `fseek`, `fclose`), no descriptor publication, no validation guard
  predicates beyond placeholder request-shape rejection, no callback storage in
  context, and no tool-only read scaffold. Those land in Phases 213/214/215/216.
- The Phase 212 boundary doctest drives the actor only through `process_event(...)`,
  inspects state via `is(state<...>)` / `visit_current_states(...)`, and asserts the
  source-text scope guardrail that no platform read primitives leak into the boundary
  headers.

Canonical refs:

- `AGENTS.md` — engineering contract (RTC actor model, no-queue invariant, allocation
  discipline, naming/style, context rules, runtime-branching rules).
- `CLAUDE.md` — project instructions overriding default behavior.
- `docs/rules/sml.rules.md` — normative SML rules (destination-first transitions,
  `sml::unexpected_event` handling, component layout, callback semantics).
- `docs/rules/cpp.rules.md` — C++ coding rules (when present).
- `src/emel/io/mmap/{context,events,errors,guards,actions,detail,sm}.hpp` — closest
  structural analog. Phase 212 mirrors layout, file order, `state_*` / `event_*` /
  `guard_*` naming idioms, and `sml::unexpected_event` coverage.
- `src/emel/io/loader/` — orchestration-only counterpart that consumes I/O events in
  later phases.
- `src/emel/model/tensor/` — residency owner; must remain unchanged in Phase 212.
- `src/emel/machines.hpp` — additive top-level alias surface (mirrors `IoMmap`
  pattern when the new `IoRead` alias is added in Phase 212).
- `tests/io/mmap/lifecycle_tests.cpp` — boundary doctest pattern reference.
- `.planning/ROADMAP.md` — v1.25 active block, Phase 212 detail section.
- `.planning/REQUIREMENTS.md` — READ-01 wording and traceability.
- `.planning/milestones/v1.24-phases/204-mmap-strategy-component-boundary/` — full
  Phase 204 closeout (CONTEXT, PLAN, SUMMARY, VERIFICATION, VALIDATION) as the
  authoritative example of a completed boundary phase.
