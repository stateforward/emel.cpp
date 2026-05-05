---
phase: 206-mapped-descriptor-errors-and-lifetime
status: verified
requirements:
  - MMAP-03
  - LIFE-01
  - ERR-01
created: 2026-05-04T17:05:00Z
last_updated: 2026-05-04T17:25:00Z
---

# Phase 206 Verification

## Source-Backed Requirement Check

### MMAP-03 — Deterministic mapped tensor buffer descriptor on success

The mmap strategy returns a deterministic descriptor on success
without owning tensor residency lifecycle and without storing
dispatch-local request data in `action::context`.

Source evidence:

- Success path in `src/emel/io/mmap/sm.hpp` walks
  `state_slot_reservation_decision` →
  `state_file_open_decision[file_open_succeeded]` →
  `state_mapping_decision[mapping_succeeded]` →
  `state_publish_done_decision`. The publish step calls
  `effect_publish_map_tensor_done` which constructs
  `events::map_tensor_done{ request, handle = reserved_slot,
  buffer = mapped_base, buffer_bytes = mapped_bytes }`.
- `events::map_tensor_done` (`src/emel/io/mmap/events.hpp`) carries
  `handle` (`uint32_t`), `buffer` (`const void*`), and
  `buffer_bytes` (`uint64_t`) — the deterministic descriptor.
- `action::context` (`src/emel/io/mmap/context.hpp`) holds only
  persistent actor-owned state (the slot pool and free stack). All
  per-dispatch request/output/status fields live in
  `detail::map_attempt_status` (`src/emel/io/mmap/detail.hpp`)
  attached to the internal runtime event for the dispatch lifetime
  only.
- Tensor residency lifecycle is not modified anywhere; `model/tensor`
  remains untouched. The mapped descriptor is published as a value
  in the done callback and is not owned beyond the slot's
  actor-internal record.

Test evidence
(`tests/io/mmap/lifecycle_tests.cpp`):

- `io mmap returns a deterministic mapped descriptor on success`
  writes a 4 KiB temp file, dispatches `map_tensor`, and verifies
  the done callback receives `handle != k_invalid_mapping_handle`,
  `buffer != nullptr`, `buffer_bytes == 4096`, and that the first
  and last mapped bytes match the file payload.
- `io mmap success records when no done callback is supplied`
  verifies the descriptor commit still occurs (the actor remains
  in `state_ready` with the slot in use) when the request omits
  `on_done`.

### LIFE-01 — Deterministic, bounded, actor-owned unmap lifetime

Mapped buffer lifetime is owned by the actor and tied to the
explicit `release_mapping` event. There is no destructor-driven
unmap and no implicit cleanup outside dispatch.

Source evidence:

- `event::release_mapping` carries a `uint32_t handle` plus optional
  `on_done`/`on_error` callbacks
  (`src/emel/io/mmap/events.hpp`).
- The release chain in `sm.hpp` is:
  `state_ready[event::release_mapping]` →
  `state_release_decision` (entry effect_begin_release) →
  `state_release_in_use_decision[handle_in_range]` →
  `state_unmap_decision[slot_in_use]` (entry
  `effect_attempt_unmap`) →
  `state_release_publish_done_decision[unmap_succeeded]` (entry
  `effect_release_slot_after_unmap`) →
  `state_release_done_callback` →
  `state_ready` (record).
- `effect_release_slot_after_unmap`
  (`src/emel/io/mmap/actions.hpp`) clears the slot and pushes the
  released index back onto the free stack, restoring it for LIFO
  reuse. `effect_mark_unmap_failed_and_release_slot` releases the
  bookkeeping even when the platform unmap reports failure so the
  actor never leaks a slot.
- `slot::os_resource` records the open file descriptor (or
  `HANDLE`) for the lifetime of the mapping; the platform unmap
  helper closes it.

Test evidence:

- `io mmap release happy path returns slot to the free pool`
  verifies the round trip and confirms LIFO slot reuse on a
  subsequent map. Both maps land on the same slot index.
- `io mmap release rejects out-of-range handle` verifies validation
  fail-closed for handles outside `[0, k_max_mappings)`.
- `io mmap release rejects double release on the same handle`
  verifies the actor refuses to unmap a slot it does not own and
  publishes `error::invalid_request` rather than calling the
  platform unmap a second time.
- `io mmap fails closed without an error callback` exercises the
  no-callback recovery for both map and release dispatches.

### ERR-01 — Deterministic error categories with no exceptions across boundaries

Failures are surfaced as distinct error categories through
`events::map_tensor_error` or `events::release_mapping_error`;
no exception crosses any actor or API boundary.

Source evidence:

- `errors.hpp` enumerates: `none`, `invalid_request`,
  `unsupported_platform`, `unsupported_resource`,
  `resource_exhausted`, `file_open_failed`, `mapping_failed`,
  `unmap_failed`, `internal_error`.
- Each error category routes through a dedicated decision state
  with a dedicated mark-effect:
  `state_invalid_request_error_decision /
  effect_mark_invalid_request`,
  `state_unsupported_resource_error_decision /
  effect_mark_unsupported_{file,offset,length,layout}`,
  `state_unsupported_platform_error_decision /
  effect_mark_unsupported_platform`,
  `state_resource_exhausted_error_decision /
  effect_mark_resource_exhausted`,
  `state_file_open_failed_error_decision /
  effect_release_reserved_slot_on_open_failure`,
  `state_mapping_failed_error_decision /
  effect_close_open_resource_and_release_slot_on_mapping_failure`,
  `state_release_invalid_handle_error_decision /
  effect_mark_release_invalid_handle`,
  `state_unmap_failed_error_decision /
  effect_mark_unmap_failed_and_release_slot`.
- All effects, guards, helpers, and platform OS-call helpers in
  `actions.cpp` are `noexcept`. No `try`/`throw`/`catch` appears in
  any io/mmap source.

Test evidence: each error category is exercised through
`process_event(...)` with the published callback's `err` field
asserted equal to the expected `error::*` value:

- `invalid_request` — empty file_path, zero byte_size, double release.
- `unsupported_resource` — file_index above bound, unaligned offset,
  byte_size above bound, layout overflow.
- `unsupported_platform` — covered structurally on hosts where the
  platform gate evaluates false (Phase 205 contract preserved).
- `resource_exhausted` — slot pool drained by 256 successful maps.
- `file_open_failed` — non-existent path with all preconditions
  satisfied; slot pool returns to full free count after recovery.
- `mapping_failed` — directory path mapped; the actor surfaces
  either `mapping_failed` (mmap rejects directory fd) or
  `file_open_failed` (platforms that reject directory open up
  front), depending on platform.
- `unmap_failed` — uncovered structurally; reachable only when the
  platform helper reports failure. Coverage gap recorded for Phase
  209.

## Out-of-Scope Verification

Phase 206 did NOT introduce or modify:

- Any change to `model/tensor`, `model/loader`, benchmark,
  paritychecker, or embedded probe code (verified by reading
  `git status` and the changed-files scope).
- Any new C++ template, exception, or dynamic dispatch on the hot
  path. The actor still uses compile-time SML transitions and
  `noexcept` actions/guards throughout.
- Any heap allocation during dispatch. The slot pool, free stack,
  and per-dispatch carriers are all stack-resident or
  `std::array`-resident inside the actor; tests rely on
  `std::filesystem` and `std::string` only at test scope.
- Any platform header in `actions.hpp`, `detail.hpp`, `guards.hpp`,
  `sm.hpp`, `events.hpp`, `context.hpp`, or `errors.hpp`. The
  boundary-source test enforces this directly.
- Any cooperative async, staged read/copy, device strategy, or
  tool-only mmap scaffold (out of scope for v1.24).

## Result

Phase 206 implements MMAP-03, LIFE-01, and ERR-01 against the
canonical `emel::io::mmap` Stateforward.SML actor with source-backed
evidence, real `mmap`/`munmap`/`open`/`close` plumbing through the
maintained codepath, deterministic error categorisation, no
exceptions, no hot-path allocation, no test-only scaffolds, and no
override applied to the changed-file scoped quality gate.
