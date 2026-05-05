---
phase: 207-tensor-owned-mmap-integration
status: in_progress
requirements:
  - TIO-01
  - TIO-02
created: 2026-05-04T17:35:00Z
---

# Phase 207 Context

Phase 207 wires `model/tensor` to the public `emel/io/mmap` event surface
introduced in Phase 206. Tensor receives two new public events
(`request_mapped_load`, `release_mapped_load`); each translates into a
synchronous cross-actor `process_event(...)` call against an injected
`emel::io::mmap::sm*`. Tensor remains the owner of bind, evict, and
residency lifecycle transitions; io/mmap remains the owner of the
mapping resource and unmap call. Public runtime/loader/benchmark/parity
surfaces are unchanged and remain deferred to Phases 208 and 210.

Locked decisions:

- Tensor receives the io/mmap dispatcher through a single context
  aggregate field: `action::context::io_mmap`, a non-owning
  `emel::io::mmap::sm*` defaulted to `nullptr`. Existing call sites
  default-construct the context exactly as before; only call sites that
  want the new flow inject a pointer to a long-lived io/mmap actor.
- `tensor_storage` gains a `mapping_handle` column
  (`std::array<uint32_t, max_tensors>`) defaulted to
  `emel::io::mmap::k_invalid_mapping_handle`. A tensor with
  `mapping_handle != k_invalid_mapping_handle` is mmap-loaded and is
  the only valid target of `release_mapped_load`.
- The two new public events follow Phase 205+206 callback patterns:
  optional `on_done`/`on_error` `emel::callback`s, no `cmd_*` prefix,
  `_done`/`_error` outcome events with explicit suffixes. The
  `file_path` field on `event::request_mapped_load` is a
  `std::string_view`. The Phase 206 caller-owned-null-terminated
  contract applies and is documented again here.
- Tensor never calls a low-level platform mmap API. It only
  constructs `emel::io::mmap::event::map_tensor` /
  `emel::io::mmap::event::release_mapping` payloads and dispatches
  them via `ctx.io_mmap->process_event(...)`. The io/mmap callbacks
  fire synchronously inside the io/mmap dispatch and capture the
  outcome into the tensor's per-dispatch
  `request_mapped_load_status` / `release_mapped_load_status` carrier
  attached to the tensor's internal runtime event. Tensor's own state
  machine then routes success vs. failure via explicit guards on the
  captured fields.
- The io/mmap callbacks NEVER call back into
  `tensor_sm.process_event(...)` (no re-entrancy). They only mutate
  fields in the tensor's per-dispatch status struct. This is enforced
  in code review, not at runtime.
- `request_mapped_load` request validation (separate explicit guards
  before any cross-actor dispatch): `tensor_id` in
  `[0, active_extent)`, `file_path` non-empty, `byte_size > 0`,
  tensor lifecycle currently `unbound` or `evicted` (rejecting
  `resident` to avoid double-mapping a tensor without an explicit
  release), and `ctx.io_mmap != nullptr`. Each precondition has its
  own decision state and transition.
- `release_mapped_load` request validation: `tensor_id` in range,
  `mapping_handle != k_invalid_mapping_handle` (rejecting
  release-on-non-mmap-loaded tensors), and `ctx.io_mmap != nullptr`.
- Outcome semantics: the io/mmap result categories
  (`unsupported_platform`, `unsupported_resource`,
  `resource_exhausted`, `file_open_failed`, `mapping_failed`,
  `unmap_failed`, `internal_error`) are surfaced verbatim through the
  tensor's `_error` events as `emel::error::type`. Tensor does not
  re-categorise io/mmap errors.
- No mirrored status fields, no action-selected callbacks, no
  context phase flags. All routing is via guards and explicit
  transitions.
- Scope strictly tensor + io/mmap. `model/loader` is not modified.
  `src/emel/machines.hpp` is not modified. Benchmark, paritychecker,
  embedded probe, and any tool surface are not modified.
- The Phase 204 transitional bench-regression override is not
  consumed.

Canonical refs:

- `docs/rules/sml.rules.md`
- `AGENTS.md`
- `src/emel/io/mmap/`
- `src/emel/model/tensor/`
- `.planning/milestones/v1.24-phases/204-mmap-strategy-component-boundary/`
- `.planning/milestones/v1.24-phases/205-mmap-validation-platform-gating/`
- `.planning/milestones/v1.24-phases/206-mapped-descriptor-errors-and-lifetime/`
- `.planning/ROADMAP.md` (v1.24 active)
- `.planning/REQUIREMENTS.md` (v1.24)
