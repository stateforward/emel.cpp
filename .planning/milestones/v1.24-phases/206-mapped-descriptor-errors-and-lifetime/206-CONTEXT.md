---
phase: 206-mapped-descriptor-errors-and-lifetime
status: in_progress
requirements:
  - MMAP-03
  - LIFE-01
  - ERR-01
created: 2026-05-04T17:05:00Z
---

# Phase 206 Context

Phase 206 lands the first end-to-end mapping path inside the
`emel::io::mmap` Stateforward.SML strategy actor. The actor now performs
real platform-level `open`/`mmap`/`munmap`/`close` calls, returns a
deterministic mapped tensor buffer descriptor on success, and owns the
release/unmap lifetime via an explicit public event. Tensor-to-I/O wiring,
public runtime/loader/benchmark exposure, and publication remain deferred
to Phases 207, 208, and 210.

Locked decisions (per main directive 2026-05-04):

- **File identification:** `event::map_tensor_request` carries
  `std::string_view file_path`, caller-owned for the duration of dispatch.
  An explicit `file_path_non_empty` guard rejects empty paths before any
  platform attempt. The platform call consumes `file_path.data()`, so the
  Phase 206 caller contract is "the storage backing `file_path` MUST be
  null-terminated for the duration of dispatch". This is documented in
  `events.hpp` and exercised by tests through `std::string` storage whose
  `c_str()` contract guarantees null termination. Phase 207 may add a
  separate `register_file` event later for open-once amortization.
- **Slot ownership:** Mapped resources live in an actor-owned fixed-capacity
  slot table inside `action::context`. Storage is a
  `std::array<slot, k_max_mappings>` (k_max_mappings = 256 by default)
  with a deterministic LIFO free-stack (`free_stack[k_max_mappings]` plus
  `free_count`) initialised once in the context default constructor. No
  heap allocation occurs during dispatch. Slot reservation is a
  free-stack pop performed by an action; the binary "is a slot available"
  decision is an explicit guard.
- **Release surface:** `event::release_mapping` carries a `uint32_t handle`
  plus optional `on_done`/`on_error` callbacks. Outcome events
  `events::release_mapping_done` and `events::release_mapping_error` are
  added. Release validates `release_handle_in_range` and
  `release_handle_in_use` with explicit guards before any unmap. Callbacks
  fire synchronously and are never stored.
- **Error taxonomy (ERR-01):** Distinct categorical errors:
  `invalid_request`, `unsupported_platform`, `unsupported_resource`,
  `resource_exhausted`, `file_open_failed`, `mapping_failed`,
  `unmap_failed`, `internal_error`. Each error category maps to a
  dedicated decision/error state and a dedicated mark-effect.
- **File layout:** Platform-specific OS calls live entirely in
  `src/emel/io/mmap/actions.cpp` behind compile-time `#if defined(_WIN32)`
  selection. `actions.hpp` declares the OS-touching effect operator()
  signatures; their bodies are defined out-of-line in `actions.cpp`. No
  new file names beyond canonical bases. `detail.hpp` stays data-only
  (per-dispatch carriers, slot record types). `errors.hpp`,
  `guards.hpp`, `events.hpp`, `context.hpp`, `sm.hpp` follow the existing
  header-only pattern. `EMEL_IO_MMAP_PLATFORM_SUPPORTED` flips to `1`
  when the host is POSIX or `_WIN32`.
- **OS-call placement:** Per main constraint 6, an action performs the
  already-selected OS attempt and records its raw result into the
  per-dispatch runtime carrier. The next state routes success vs. failure
  through explicit guards on the recorded result. No action chooses
  behaviour, no detail helper chooses behaviour, no exception crosses an
  actor boundary. The constraint
  "ALWAYS keep actions bounded and non-blocking during dispatch" is
  applied as bounded; one-shot loader-setup syscalls (`open`/`mmap`/
  `munmap`/`close`) are accepted under the same precedent that allows
  one-time initialisation work outside hot inference paths. This is
  recorded as a deliberate Phase 206 trade-off.
- **Context discipline:** Per AGENTS, context holds only persistent
  actor-owned state (the slot pool). All per-dispatch fields (open fd,
  mmap base, attempt success flags, reservation index, target handle)
  live in `detail::map_attempt_status` / `detail::release_attempt_status`
  attached to the internal runtime event for the dispatch lifetime only.
- **No external surface change beyond io/mmap:** `model/loader`,
  `model/tensor`, benchmark, paritychecker, and embedded probe code is
  untouched. `src/emel/machines.hpp` keeps its existing
  `emel::IoMmap = emel::io::mmap::sm` alias and is not modified by
  Phase 206.
- **No deferred work absorbed:** Phase 207's tensor-owned mmap
  integration, Phase 208's public runtime exposure, and Phase 210's
  publication artefacts remain deferred. The Phase 204 transitional
  bench-regression override is not consumed.

Canonical refs:

- `docs/rules/sml.rules.md`
- `AGENTS.md`
- `src/emel/io/mmap/`
- `src/emel/io/loader/`
- `.planning/milestones/v1.24-phases/204-mmap-strategy-component-boundary/`
- `.planning/milestones/v1.24-phases/205-mmap-validation-platform-gating/`
- `.planning/ROADMAP.md` (v1.24 active)
- `.planning/REQUIREMENTS.md` (v1.24)
