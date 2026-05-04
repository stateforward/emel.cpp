---
phase: 205-mmap-validation-platform-gating
status: in_progress
requirements:
  - MMAP-02
  - PLAT-01
created: 2026-05-04T16:30:00Z
---

# Phase 205 Context

Phase 205 adds explicit precondition validation and platform gating to the
`emel::io::mmap` Stateforward.SML strategy actor introduced in Phase 204. The
phase MUST NOT perform real `mmap`/`munmap` calls, publish a mapped descriptor,
introduce mapped-buffer lifetime semantics, or wire tensor integration. Concrete
mapped descriptors, lifetime, errors, tensor integration, and public runtime
exposure are owned by Phases 206, 207, and 208.

Locked decisions:

- Validation is modelled as an explicit chain of decision states inside
  `emel::io::mmap::sm`. Each precondition (`request`, `file`, `offset`,
  `length`, `layout`, `platform`) has a dedicated decision state with
  guarded completion transitions. Preconditions are accepted or rejected
  before any mapping attempt is even structurally reachable.
- Validation guards are pure predicates over `(detail::map_tensor_runtime,
  context)` with no side effects, no allocation, and no behavior selection
  that is not modelled in `sm.hpp`.
- Platform support is a fail-closed compile-time gate exposed as the
  `EMEL_IO_MMAP_PLATFORM_SUPPORTED` macro (default `0`). Phase 206 will flip
  the gate per platform when concrete mapping lands. The gate is consumed by
  the `platform_mmap_unsupported` guard inside `guards.hpp` only.
- Validation bound constants (`k_max_file_index`,
  `k_required_offset_alignment`, `k_max_mapping_bytes`) live alongside the
  mmap error enum in `errors.hpp` so guards can reference them without a new
  module surface.
- The validation success branch (all preconditions satisfied AND platform
  supported) is intentionally omitted in Phase 205 because no concrete
  mapping exists. The platform-supported branch is statically unreachable on
  shipped builds because `EMEL_IO_MMAP_PLATFORM_SUPPORTED` is `0` everywhere
  in this phase. Phase 206 introduces the supported-platform completion
  destination together with the mapped descriptor and lifetime contract.
- No new public API surface is added in Phase 205. The `event::map_tensor`,
  `event::map_tensor_request`, `events::map_tensor_done`, and
  `events::map_tensor_error` shapes from Phase 204 are unchanged.
- Context remains an empty struct. Per-dispatch validation outcome is carried
  in the existing private `detail::runtime_status` reference attached to the
  internal `detail::map_tensor_runtime` event, never in machine context.
- No `std::mutex`, `std::thread`, `std::filesystem`, `<unistd.h>`, or
  platform mapping headers are introduced. Tests MAY use `std::filesystem`
  only for source-file boundary inspection (already in Phase 204 tests).
- `model/loader`, benchmark, paritychecker, and embedded probe surfaces are
  untouched in Phase 205. Tensor-to-I/O integration remains deferred to
  Phase 207.

Canonical refs:

- `docs/rules/sml.rules.md`
- `AGENTS.md`
- `src/emel/gbnf`
- `src/emel/io/mmap/`
- `.planning/milestones/v1.24-phases/204-mmap-strategy-component-boundary/`
- `.planning/ROADMAP.md` (v1.24 active)
- `.planning/REQUIREMENTS.md` (v1.24)
