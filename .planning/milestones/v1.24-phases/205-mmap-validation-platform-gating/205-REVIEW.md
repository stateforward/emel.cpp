---
phase: 205-mmap-validation-platform-gating
status: clean
reviewed_phases:
  - 204
  - 205
reviewed_paths:
  - src/emel/io/mmap/context.hpp
  - src/emel/io/mmap/detail.hpp
  - src/emel/io/mmap/events.hpp
  - src/emel/io/mmap/errors.hpp
  - src/emel/io/mmap/guards.hpp
  - src/emel/io/mmap/actions.hpp
  - src/emel/io/mmap/sm.hpp
  - tests/io/mmap/lifecycle_tests.cpp
  - src/emel/machines.hpp (alias only)
  - CMakeLists.txt (test entry only)
created: 2026-05-04T17:00:00Z
last_updated: 2026-05-04T17:00:00Z
---

# Phase 205 Code Review

## Scope

Autonomous review of the Phase 204 boundary plus the Phase 205 validation
and platform-gating implementation in `emel::io::mmap` and the matching
test suite, prior to Phase 206 planning. No edits made.

## Verdict

**Clean.** No blockers, no must-fix bugs, no AGENTS/SML rule violations,
and Phase 206 deferrals are correctly absent. A small number of
informational observations are listed below for the Phase 206 author.

## AGENTS / SML Rule Compliance

### RTC actor model and no-queue invariant

- `sm` exposes a single public dispatch entrypoint
  (`process_event(event::map_tensor)`) plus inherited
  `process_event(unrelated_event)` for unexpected handling. No queues,
  no posts-for-later. ✓
- Internal phase progress uses `sml::completion<detail::map_tensor_runtime>`
  with a small, statically bounded chain of decision states (≤ 7 phase
  hops per top-level dispatch). ✓
- `sml::unexpected_event<sml::_>` handlers are declared for every
  reachable state (`state_ready`, `state_request_decision`,
  `state_file_decision`, `state_offset_decision`,
  `state_length_decision`, `state_layout_decision`,
  `state_platform_decision`,
  `state_invalid_request_error_decision`,
  `state_unsupported_resource_error_decision`,
  `state_unsupported_platform_error_decision`,
  `state_error_callback`). ✓
- No actor-internal re-entrancy, no shared models, no cross-actor calls.
  ✓

### Transition table style

- Destination-first rows (`sml::state<dst> <= src + event [guard] / action`)
  throughout. ✓
- Destination state and `<=` on the same line for every row. ✓
- Leading-comma row style after the first row inside
  `make_transition_table(...)`. ✓
- Phase-label divider comments separate
  Acceptance / Request / File / Offset / Length / Layout / Platform /
  Error publication / Unexpected sections. ✓
- `// clang-format off/on` is narrowly scoped around the table only. ✓

### Behavior selection

- All routing decisions live in `guards.hpp` predicates consumed by
  `sm.hpp`. ✓
- `actions.hpp` effects only mark the per-dispatch `runtime_status` and
  invoke the user-provided error callback; no `if`/`else if`/`switch`/
  `?:`/loop-as-branch constructs. The `effect_on_unexpected` template
  uses `if constexpr (requires { ev.ctx.err; })` which is a compile-time
  conditional explicitly allowed by the rules. ✓
- `detail.hpp` contains only data carriers (`runtime_status`,
  `map_tensor_runtime`); no helpers, no routing, no support probing. ✓
- `errors.hpp` defines categorical error codes plus validation bound
  constants only; no logic. ✓

### Allocation and dispatch

- All actions and guards are `noexcept`. ✓
- `sm::process_event(event::map_tensor)` constructs a stack-resident
  `detail::runtime_status` and `detail::map_tensor_runtime` per dispatch
  before calling `base_type::process_event`; no heap allocation. ✓
- No mutex, no sleep, no I/O wait, no wall-clock read in any guard or
  action. ✓
- Tracing is absent (component-local). ✓

### Events, outcomes, errors

- `event::map_tensor` and `event::map_tensor_request` are noun-shaped
  trigger intents. Outcome events (`events::map_tensor_done`,
  `events::map_tensor_error`) carry explicit `_done` / `_error` suffixes.
  ✓
- No `cmd_*` prefixed events. ✓
- Required event payload fields (`request`) use `const T&`; the optional
  `on_done` and `on_error` callbacks use `emel::callback`. No owning
  pointers in events, no dynamic containers. ✓
- Internal-only `detail::map_tensor_runtime` carries a mutable
  `runtime_status &` reference for synchronous same-RTC handoff and is
  not exposed via public outcome event payloads. ✓
- Failures are modeled via explicit error decision states and an
  explicit `state_error_callback` publication state; no synthetic
  fault-injection knobs, no test-only control fields. ✓

### Context rules

- `action::context` is an empty struct. ✓ (Matches the AGENTS rule that
  "if a machine has no persistent actor-owned state, context MUST be an
  empty struct.")
- No dispatch-local data (request pointers, phase flags, step indexes,
  status codes) is mirrored into context. ✓
- Per-dispatch carrier (`runtime_status`) lives on the
  `process_event(map_tensor)` stack frame and is referenced through the
  internal event only. ✓

### Naming

- States use `state_*`. Effects use `effect_*`. Guards live in the
  `guard::` namespace with semantic predicate names matching the
  established sibling-component convention used by `src/emel/io/loader`
  (e.g. `tensor_span_valid`, `strategy_mapped_file`). The
  AGENTS prefix policy is applied semantically; the namespace is the
  prefix in the io/loader and io/mmap families. ✓
- Constants use `k_` snake_case. ✓
- Types are lower_snake_case for non-exported internal types. ✓

### Domain and platform isolation

- No platform mapping calls
  (`mmap`, `munmap`, `CreateFileMapping`, `MapViewOfFile`, `pread`,
  `std::ifstream`) appear in `actions.hpp`, `detail.hpp`, `guards.hpp`,
  or `sm.hpp`; the boundary-source test asserts this. ✓
- No platform headers are included from any io/mmap source. ✓
- The single platform knob is the compile-time macro
  `EMEL_IO_MMAP_PLATFORM_SUPPORTED` (default `0`) consumed only by the
  `platform_mmap_supported` / `platform_mmap_unsupported` guards. ✓
- No leakage into `model/loader`, `model/tensor`, benchmark, or
  paritychecker code. ✓

## Behavioral Bug Scan

Walked every reachable path in the transition table and traced the
test-driven scenarios. No incorrect routing, no missing
unexpected-event handler, no mutually-non-exhaustive guard pair, no
unhandled completion event found.

Observations on edge-case math (all correct):

- `layout_supported` checks `offset <= UINT64_MAX - size`. Reachable only
  after `length_within_bounds` accepted `size <= k_max_mapping_bytes ==
  (1ULL << 40)` and `request_span_valid` accepted `size > 0`, so the
  subtraction `UINT64_MAX - size` is well-defined and never wraps. The
  predicate correctly catches address-space wraparound for the
  `0xFFFFFF0000000000`-style overflow case exercised by
  `io mmap rejects layouts that overflow the address space`.
- `offset_aligned` uses `% k_required_offset_alignment` with
  `k_required_offset_alignment = 4096u`. Page-aligned offsets like
  `0`, `4096`, `8192` pass; the existing recovery test
  (`file_offset = 8192u`, `byte_size = 64u`) reaches the platform gate
  exactly as before, preserving Phase 204 behavior.
- `file_index_valid` allows up to `k_max_file_index = 65534u`, leaving
  `65535` (the natural `uint16_t` sentinel) as the canonical
  unsupported sentinel. The new test
  `io mmap rejects out-of-range file_index as unsupported resource`
  drives this with `k_max_file_index + 1u`.
- `effect_on_unexpected` is generic over event type and guards its
  mutation with `if constexpr (requires { ev.ctx.err; })`. Sentinel
  `unrelated_event` lacks `.ctx.err` so the empty constexpr branch
  fires; the runtime-payload branch (lines 85-86) is dead in current
  tests but reachable by future tests that inject a stale
  `detail::map_tensor_runtime` event. Coverage of that span is owned
  by VAL-01 / Phase 209.

## Test Coverage Scan

`tests/io/mmap/lifecycle_tests.cpp` exercises:

- Canonical aliases (`emel::io::mmap::sm`, `emel::IoMmap`).
- `request_span_invalid` / `effect_mark_invalid_request` →
  `error::invalid_request`.
- `file_index_invalid` / `effect_mark_unsupported_file` →
  `error::unsupported_resource`.
- `offset_unaligned` / `effect_mark_unsupported_offset` →
  `error::unsupported_resource`.
- `length_overflow` / `effect_mark_unsupported_length` →
  `error::unsupported_resource`.
- `layout_unsupported` / `effect_mark_unsupported_layout` →
  `error::unsupported_resource`.
- `platform_mmap_unsupported` / `effect_mark_unsupported_platform` →
  `error::unsupported_platform` (preconditions-pass scenario plus the
  Phase 204 recovery scenario).
- Fail-closed without an error callback (no callback invocation, return
  to `state_ready`).
- Fail-closed dispatch sequencing and recovery to `state_ready`.
- Unexpected event before and after a normal dispatch.
- Boundary-source assertion that no concrete platform-mapping
  identifiers appear in actions/detail/guards/sm and that the new
  decision state names are present in `sm.hpp`.

All tests drive the strategy through `sm::process_event(map_tensor)`
and inspect state via `is(...)` and the published callback payload, as
required by AGENTS for behavior tests.

Coverage gaps that are correctly deferred:

- Boundary tests at exact bound values
  (`file_index = k_max_file_index`,
  `byte_size = k_max_mapping_bytes`) are absent. These are nice-to-have
  edge-strengtheners but not required by MMAP-02 wording. Suggest
  rolling them into Phase 209's behavior-test sweep.
- The `effect_on_unexpected` payload-bearing branch (actions.hpp lines
  85-86) is not driven; this is owned by Phase 209 (VAL-01).
- The supported-platform completion destination is intentionally
  unreachable in Phase 205 (gate is `0`). Phase 206 introduces both the
  destination and the corresponding success-path tests.

## Phase 206 Deferral Verification

Phase 205 correctly does NOT contain:

- Real `mmap`/`munmap` or platform-specific mapping calls.
- A mapped descriptor success state out of `state_platform_decision`.
- A `state_preconditions_validated`, `state_mapping_decision`, or any
  successor that produces a buffer.
- A populated `events::map_tensor_done` payload (the Phase 204 default
  shape with `buffer = nullptr`, `buffer_bytes = 0u` is unchanged).
- An unmap or lifetime-bound resource handle.
- Tensor-to-I/O event surfaces or `model/tensor` integration.
- Any change to `model/loader`, benchmark, paritychecker, or embedded
  probe surfaces.

These deferrals match Phase 205 scope and the v1.24 ROADMAP.

## Informational Observations (non-blocking)

1. `sm` inherits `using base_type::process_event`, which makes the
   internal `process_event(detail::map_tensor_runtime)` overload
   reachable by external callers in addition to the
   `event::map_tensor` overload. This is needed for the
   unexpected-event test path, but a future tightening could private
   the runtime overload behind a friend-only seam if direct dispatch
   becomes a real concern. Not a Phase 205 issue and not a rule
   violation today (`detail::map_tensor_runtime` lives in the
   `detail::` namespace which is the established
   non-public marker repo-wide).

2. `effect_mark_unsupported_file/offset/length/layout` collapse into
   `error::unsupported_resource`. This is correct under PLAT-01 (one
   "fail closed on unsupported file/resource shapes" category) but
   loses per-precondition diagnostic granularity. If Phase 206 / ERR-01
   needs richer error categories for diagnostics, that work is the
   right place to revisit. No action needed in Phase 205.

3. `platform_mmap_unsupported{}(ev, ctx)` calls
   `platform_mmap_supported{}(ev, ctx)` and negates. Functionally fine
   and matches the established `_invalid` paired-guard convention from
   io/loader. Consider also exposing a class-level `static constexpr
   bool` for compile-time consumers in Phase 206; not needed for the
   current `if constexpr` body in the guard itself.

4. `events::map_tensor_done` and `events::map_tensor_error` carry a
   `const event::map_tensor &request` back-pointer. This is allowed by
   the optional-correlation rule in AGENTS (same-RTC handoff). When
   Phase 206 wires the success path, double-check that no caller stores
   either outcome event past dispatch return.

5. Phase 204's transitional `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION=1`
   was NOT consumed by Phase 205 (no benchmark-affecting changed files
   this phase). The Phase 204 carry-forward is still owed at Phase 210
   closeout.

## Status

Clean. Phase 206 may proceed without rework of any Phase 204+205 io/mmap
artifact.
