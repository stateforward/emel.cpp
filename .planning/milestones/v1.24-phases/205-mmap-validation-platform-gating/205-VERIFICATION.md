---
phase: 205-mmap-validation-platform-gating
status: verified
requirements:
  - MMAP-02
  - PLAT-01
created: 2026-05-04T16:30:00Z
last_updated: 2026-05-04T16:55:00Z
---

# Phase 205 Verification

## Source-Backed Requirement Check

### MMAP-02 — Validation through explicit guards and transitions

The mmap strategy validates `request`, `file`, `offset`, `length`, `layout`,
and `platform` preconditions via dedicated decision states with guarded
completion transitions before any mapping attempt is structurally reachable.

Source evidence (`src/emel/io/mmap/sm.hpp`):

- `state_request_decision` decides on `request_span_valid` vs
  `request_span_invalid`, routing failures to
  `state_invalid_request_error_decision` via
  `effect_mark_invalid_request`.
- `state_file_decision` decides on `file_index_valid` vs
  `file_index_invalid`, routing failures to
  `state_unsupported_resource_error_decision` via
  `effect_mark_unsupported_file`.
- `state_offset_decision` decides on `offset_aligned` vs `offset_unaligned`
  using `k_required_offset_alignment`, routing failures to
  `state_unsupported_resource_error_decision` via
  `effect_mark_unsupported_offset`.
- `state_length_decision` decides on `length_within_bounds` vs
  `length_overflow` using `k_max_mapping_bytes`, routing failures to
  `state_unsupported_resource_error_decision` via
  `effect_mark_unsupported_length`.
- `state_layout_decision` decides on `layout_supported` vs
  `layout_unsupported` (no-wraparound predicate over
  `file_offset + byte_size`), routing failures to
  `state_unsupported_resource_error_decision` via
  `effect_mark_unsupported_layout`.
- `state_platform_decision` decides on `platform_mmap_unsupported`
  (compile-time gate from `EMEL_IO_MMAP_PLATFORM_SUPPORTED`), routing the
  fail-closed case to `state_unsupported_platform_error_decision` via
  `effect_mark_unsupported_platform`.

All guards are pure predicates over `(detail::map_tensor_runtime,
action::context)` defined in `src/emel/io/mmap/guards.hpp`. No behavior
selection lives in `actions.hpp` or `detail.hpp`; every routing decision is
a guard in `guards.hpp` consumed by `sm.hpp`.

Test evidence (`tests/io/mmap/lifecycle_tests.cpp`):

- `io mmap rejects invalid request spans before any mapping attempt`
  exercises `request_span_invalid` → `error::invalid_request`.
- `io mmap rejects out-of-range file_index as unsupported resource`
  exercises `file_index_invalid` → `error::unsupported_resource`.
- `io mmap rejects unaligned file_offset as unsupported resource`
  exercises `offset_unaligned` → `error::unsupported_resource`.
- `io mmap rejects byte_size above maximum as unsupported resource`
  exercises `length_overflow` → `error::unsupported_resource`.
- `io mmap rejects layouts that overflow the address space` exercises
  `layout_unsupported` → `error::unsupported_resource`.
- `io mmap fails closed at platform gate when preconditions pass`
  exercises the all-preconditions-passing path through to the platform
  gate.

All cases drive the strategy through the public `process_event(map_tensor)`
entry point and verify final state via `is(...)` plus the published
`map_tensor_error` callback, satisfying MMAP-02.

### PLAT-01 — Platform details hidden, fail closed on unsupported platforms or shapes

Platform-specific mapping details remain entirely behind the I/O
abstraction boundary in Phase 205:

- The single platform knob is the compile-time macro
  `EMEL_IO_MMAP_PLATFORM_SUPPORTED` defined in
  `src/emel/io/mmap/errors.hpp` with a default of `0`.
- The `platform_mmap_supported` and `platform_mmap_unsupported` guards in
  `src/emel/io/mmap/guards.hpp` are the only consumers of the macro and
  expose only a boolean.
- No mmap/munmap/`CreateFileMapping`/`MapViewOfFile`/`pread`/`std::ifstream`
  identifiers appear in `actions.hpp`, `detail.hpp`, `sm.hpp`, or
  `guards.hpp`. The boundary-source test
  (`io mmap boundary contains no concrete platform mapping calls`) asserts
  this directly.
- The strategy fail-closes on every unsupported platform
  (`error::unsupported_platform` via
  `effect_mark_unsupported_platform`) and every unsupported file/resource
  shape (`error::unsupported_resource` via the `effect_mark_unsupported_*`
  family).
- No public API surface, no platform headers, and no I/O calls were
  introduced. Tests confirm the strategy returns to `state_ready` after
  every fail-closed dispatch.

This satisfies PLAT-01 within Phase 205's scope. The supported-platform
branch of `state_platform_decision` is intentionally absent until Phase 206
introduces the mapped descriptor and lifetime contract.

## Out-of-Scope Verification

Phase 205 did NOT introduce or modify:

- Real `mmap`/`munmap` calls or platform mapping headers (verified by the
  boundary-source test).
- Mapped descriptor publication, mapped buffer ownership, or unmap
  lifetime semantics (deferred to Phase 206; `events::map_tensor_done`
  payload shape unchanged from Phase 204).
- Tensor-to-I/O integration, including any `model/tensor` event surface
  for mmap selection (deferred to Phase 207).
- `model/loader`, benchmark, paritychecker, or embedded probe surfaces
  (deferred to Phase 208).
- Public docs or snapshot publication beyond regenerating maintained
  outputs affected by the implementation (deferred to Phase 210).
- Staged read/copy, device-specific, cooperative async, model-family
  widening, loader-owned byte access, or tool-only mmap scaffolds (out of
  scope for v1.24).

## Result

Phase 205 implements MMAP-02 and PLAT-01 against the canonical
`emel::io::mmap` Stateforward.SML actor with source-backed evidence and
maintained-test coverage, no real mmap calls, no public API growth, and no
override applied to the changed-file scoped quality gate.
