---
phase: 207-tensor-owned-mmap-integration
status: verified
requirements:
  - TIO-01
  - TIO-02
created: 2026-05-04T17:35:00Z
last_updated: 2026-05-04T18:40:00Z
---

# Phase 207 Verification

## Source-Backed Requirement Check

### TIO-01 — model/tensor can request mmap-backed loading through the public emel/io boundary while remaining the owner of tensor load, bind, evict, and residency transitions

`model/tensor` exposes a new public event surface that translates into
public io/mmap events through `process_event(...)` against an
injected `emel::io::mmap::sm*`. Tensor retains all bind, evict, and
residency lifecycle ownership.

Source evidence:

- `event::request_mapped_load` (`src/emel/model/tensor/events.hpp`)
  carries `tensor_id`, `file_path`, `file_offset`, `byte_size`, plus
  optional `on_done`/`on_error` callbacks. `event::release_mapped_load`
  carries `tensor_id` and the `mapping_handle` returned from a prior
  `events::request_mapped_load_done`.
- `action::context::io_mmap` (`src/emel/model/tensor/context.hpp`) is
  the only cross-actor dependency held by tensor — a non-owning
  `emel::io::mmap::sm*` injected via the new
  `sm(emel::io::mmap::sm*)` constructor.
- `effect_attempt_request_mapped_load_dispatch` and
  `effect_attempt_release_mapped_load_dispatch` (in
  `src/emel/model/tensor/actions.hpp`) construct
  `emel::io::mmap::event::map_tensor` /
  `emel::io::mmap::event::release_mapping` payloads and dispatch
  through the public `ctx.io_mmap->process_event(...)` entry. No
  low-level mmap/munmap/`MapViewOfFile`/`CreateFileMapping`/
  `<sys/mman.h>`/`<windows.h>` reference appears anywhere in
  `src/emel/model/tensor/`.
- The success path (`effect_commit_request_mapped_load`) sets
  `lifecycle = mmap_resident` and caches the buffer pointer/size in
  the existing `tensor_storage` columns — the same residency
  ownership tensor already had for `lifecycle = resident` from
  `bind_tensor`. The release path (`effect_commit_release_mapped_load`)
  sets `lifecycle = evicted` and clears the buffer fields. Bind,
  evict, and capture flows from prior phases continue to work
  unchanged.
- Cross-actor traffic is exclusively via `process_event(...)`. The
  io/mmap callback trampolines (`mapped_load_callbacks::on_io_mmap_*`)
  capture results into the per-dispatch
  `request_mapped_load_status` / `release_mapped_load_status` and do
  NOT call back into `tensor_sm.process_event(...)` — no re-entrancy.

Test evidence (`tests/model/tensor/lifecycle_tests.cpp`):

- `model_tensor_request_mapped_load_dispatches_through_io_mmap`
  writes a 4 KiB temp file, dispatches `request_mapped_load` through
  `tensor_sm{&io_mmap_actor}`, verifies the done callback receives a
  non-null buffer with matching bytes and a valid mapping handle, and
  asserts `capture_tensor_state` reports
  `lifecycle_state == mmap_resident` with the expected buffer/size.
- `model_tensor_release_mapped_load_evicts_and_clears_handle` runs a
  full request → release cycle and verifies `capture_tensor_state`
  reports `lifecycle_state == evicted` with cleared buffer fields.
- `model_tensor_request_mapped_load_rejects_when_io_mmap_absent`
  validates the unsupported-io_mmap path when no dispatcher is
  injected.

### TIO-02 — outcomes are explicit `_done`/`_error` events or states, not mirrored status fields, action-selected callbacks, or context phase flags

Each outcome category is a dedicated decision/error state with a
matching public outcome event payload.

Source evidence:

- Public outcome events (`src/emel/model/tensor/events.hpp`):
  `request_mapped_load_done` carries
  `(request, mapping_handle, buffer, buffer_bytes)`,
  `request_mapped_load_error` carries
  `(request, err: tensor::error::type, io_mmap_err: io::mmap::error::type)`,
  `release_mapped_load_done` carries `(request)`, and
  `release_mapped_load_error` carries
  `(request, err, io_mmap_err)`. Errors surface the io/mmap raw
  category alongside the tensor-side classification rather than
  collapsing into a status code.
- The transition table in `src/emel/model/tensor/sm.hpp` routes each
  outcome to a dedicated decision state:
  `state_request_mapped_load_invalid_request_error_decision`,
  `state_request_mapped_load_unsupported_io_mmap_error_decision`,
  `state_request_mapped_load_already_resident_error_decision`,
  `state_request_mapped_load_io_mmap_error_decision`,
  `state_request_mapped_load_publish_done_decision`, plus the
  symmetric
  `state_release_mapped_load_invalid_request_error_decision`,
  `state_release_mapped_load_unsupported_io_mmap_error_decision`,
  `state_release_mapped_load_handle_absent_error_decision`,
  `state_release_mapped_load_io_mmap_error_decision`,
  `state_release_mapped_load_publish_done_decision`.
- Routing decisions are explicit guards on `(runtime, ctx)`
  (`src/emel/model/tensor/guards.hpp`) — never an action-selected
  callback. `request_mapped_load_io_mmap_present_*` composite guards
  are explicit named structs that call sub-guards directly, never
  `&&` expressions inside the transition table.
- No status code is mirrored from the runtime status into
  `action::context`. The per-dispatch carriers
  (`request_mapped_load_status`, `release_mapped_load_status`) live
  on the `process_event` overload's stack frame and are referenced
  through the internal runtime event only.
- Callback presence/absence is itself a guard
  (`request_mapped_load_done_callback_present` /
  `_absent`, etc.); the action only knows whether to invoke or
  no-op based on the chosen branch — it does not select between
  paths internally.

Test evidence:

- Each error category is exercised through `process_event(...)` with
  the published callback's `err` field asserted equal to the expected
  `error::*` value:
  - `tensor::error::invalid_request` (empty file_path, byte_size = 0).
  - `tensor::error::io_mmap_unsupported` (no io/mmap injected for
    request and release).
  - `tensor::error::tensor_already_resident` (second request on a
    tensor whose lifecycle is already `mmap_resident`).
  - `tensor::error::io_mmap_failed` with
    `io_mmap_err == io::mmap::error::file_open_failed` (missing file
    surfaced through the io/mmap layer verbatim).
  - `tensor::error::tensor_unmapped` (release on a tensor that has no
    prior mapping, with `mapping_handle == k_invalid_mapping_handle`).
- The success outcomes
  (`events::request_mapped_load_done`,
  `events::release_mapped_load_done`) are exercised in the happy-path
  tests with the descriptor fields and lifecycle changes asserted.

## Out-of-Scope Verification

Phase 207 did NOT introduce or modify:

- Any change to `model/loader`, benchmark, paritychecker, embedded
  probe, or `src/emel/machines.hpp`.
- Any platform mapping header, `mmap`/`munmap`/`MapViewOfFile`/
  `CreateFileMapping` reference, or any platform-specific include
  inside `model/tensor`. The boundary-source test
  (`io boundary closeout tests avoid actor internal reach-through`)
  passes after Phase 207.
- Any cooperative async, register-once-then-map-many, staged
  read/copy, device-strategy, or tool-only mmap scaffold.
- Any new C++ template, exception, or dynamic dispatch on the
  cross-actor path. All effects, guards, and trampolines are
  `noexcept`. The cross-actor `process_event(...)` is the only
  inter-actor channel.
- Any heap allocation during dispatch. Per-dispatch carriers are
  stack-resident; `tensor_storage` and the new `io_mmap` pointer are
  the only context fields.
- Any persistent mapping-handle storage in tensor context. The
  release event carries the handle from the caller, which is the
  Phase 207 design constraint.

## Result

Phase 207 implements TIO-01 and TIO-02 against the canonical
`emel::model::tensor` Stateforward.SML actor with source-backed
evidence, real io/mmap dispatch through the maintained Phase 206
public event surface, deterministic done/error categorisation, no
exceptions, no hot-path allocation, no test-only scaffolds, and no
override applied to the changed-file scoped quality gate.
