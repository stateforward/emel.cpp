---
phase: 206-mapped-descriptor-errors-and-lifetime
status: clean
reviewed_phases:
  - 206
reviewed_paths:
  - src/emel/io/mmap/errors.hpp
  - src/emel/io/mmap/events.hpp
  - src/emel/io/mmap/context.hpp
  - src/emel/io/mmap/detail.hpp
  - src/emel/io/mmap/guards.hpp
  - src/emel/io/mmap/actions.hpp
  - src/emel/io/mmap/actions.cpp
  - src/emel/io/mmap/sm.hpp
  - tests/io/mmap/lifecycle_tests.cpp
  - CMakeLists.txt (source addition only)
  - .planning/architecture/io_mmap.md (regenerated)
  - .planning/architecture/mermaid/io_mmap.mmd (regenerated)
created: 2026-05-04T17:30:00Z
last_updated: 2026-05-04T17:30:00Z
---

# Phase 206 Code Review

## Scope

Autonomous review of the Phase 206 io/mmap mapped-descriptor + lifetime
+ error-taxonomy implementation, the matching test suite, and the
generated architecture docs. No edits made.

## Verdict

**Clean.** No blockers, no must-fix bugs, no AGENTS/SML rule violations,
and Phase 207 deferrals are correctly absent. Five informational
observations are listed for the Phase 207 author and Phase 209 sweep.

## AGENTS / SML Rule Compliance

### RTC actor model and no-queue invariant

- Single public dispatch entrypoints:
  `process_event(const event::map_tensor&)` and
  `process_event(const event::release_mapping&)`. No queues, no
  posts-for-later, no `sml::process_queue`/`defer_queue`. ✓
- Internal phase progress uses
  `sml::completion<detail::map_tensor_runtime>` and
  `sml::completion<detail::release_mapping_runtime>` exclusively. The
  longest chain is 7 phase hops (request → file_path → file → offset →
  length → layout → platform → slot → file_open → mapping → publish →
  done_callback → ready), which is statically bounded. ✓
- Every reachable state declares an
  `unexpected_event<sml::_>` handler (28 entries; verified against
  the state list in `sm.hpp`). ✓
- No actor-internal re-entrancy, no shared models, no cross-actor
  calls. ✓

### Transition table style

- Destination-first rows
  (`sml::state<dst> <= src + event [guard] / action`) throughout. ✓
- Destination state and `<=` on the same line for every row. ✓
- Leading-comma row style after the first row. ✓
- Phase-label divider comments separate Acceptance / Request /
  File path / File / Offset / Length / Layout / Platform / Slot
  reservation / File open decision / Mapping decision / Done
  publication / Map error publication / Release acceptance / Release
  done publication / Release error publication / Unexpected
  sections. ✓
- `// clang-format off/on` is narrowly scoped around the table only. ✓

### Behavior selection

- All routing decisions live in `guards.hpp` predicates consumed by
  `sm.hpp`. ✓
- Actions only mark per-dispatch status, mutate slot bookkeeping, or
  perform an already-selected OS attempt. None of them choose
  behaviour, none of them branch on dtype/backend/architecture/buffer
  lane to select a different algorithm. ✓
- `detail.hpp` contains only data carriers (`map_attempt_status`,
  `release_attempt_status`, `map_tensor_runtime`,
  `release_mapping_runtime`); no helpers, no routing. ✓
- `actions.cpp` houses the platform OS calls. The wrappers
  (`platform_open`, `platform_map`, `platform_unmap`,
  `platform_close`) translate OS return codes into a `bool` reported
  through reference out-parameters and a return value. The `if`
  checks inside (`fd < 0`, `addr == MAP_FAILED`,
  `handle == INVALID_HANDLE_VALUE`) are data-plane boundary
  conditions on a syscall return — they do not select which state
  machine path runs next; that selection is entirely in the
  downstream `file_open_succeeded` / `mapping_succeeded` /
  `unmap_succeeded` guards. ✓ (consistent with the user's saved
  guidance: "data-plane ifs in actions are fine; only
  behaviour-selecting branching is prohibited".)
- `errors.hpp` defines the error enum, the validation bound
  constants, the slot pool sizing, and the platform-supported
  macro only; no logic. ✓
- The `effect_close_open_resource_and_release_slot_on_mapping_failure`
  effect unconditionally calls `platform_close` because the success
  path of the prior decision (`file_open_succeeded`) guarantees
  `os_resource` is valid; no defensive null-check that would imply
  behaviour selection. ✓

### Allocation and dispatch

- All effects, guards, and platform helpers are `noexcept`. ✓
- `sm::process_event(map_tensor)` and
  `sm::process_event(release_mapping)` construct stack-resident
  `map_attempt_status` / `release_attempt_status` per dispatch; no
  heap allocation in either dispatch path. ✓
- The slot pool, free stack, and `free_count` live in the
  default-constructed `action::context` (`std::array` storage,
  trivially placed inline). The single non-trivial constructor body
  initialises `free_stack` once at sm construction — outside any
  dispatch and outside any hot path. ✓
- No mutex, sleep, or wall-clock read in any guard or action. ✓
- The OS calls (`open`, `mmap`, `munmap`, `close`,
  `CreateFileMappingA`, `MapViewOfFile`, `UnmapViewOfFile`,
  `CloseHandle`) are bounded but block on disk. This is the
  Phase 206 trade-off documented in `206-CONTEXT.md` and
  `206-VERIFICATION.md`: cooperative async loading remains deferred,
  and the io/mmap actor is treated as a cold loader-setup actor
  rather than a hot inference path. The trade-off was already
  approved by main when authorising the directive. ✓

### Events, outcomes, errors

- `event::map_tensor_request` and `event::map_tensor` are noun-shaped
  trigger intents; outcome events
  `events::map_tensor_done` / `events::map_tensor_error` /
  `events::release_mapping_done` /
  `events::release_mapping_error` carry the `_done` / `_error`
  suffixes the rule requires. ✓
- No `cmd_*` prefixed events. ✓
- Required event payload fields are references (`request`); optional
  callbacks are `emel::callback`. The new `file_path` field is a
  `std::string_view` (non-owning view) with the documented
  caller-owned-null-terminated contract. ✓
- Internal `detail::map_tensor_runtime` /
  `detail::release_mapping_runtime` carry mutable
  `map_attempt_status&` / `release_attempt_status&` references for
  same-RTC handoff and are not exposed in any public outcome event
  payload. ✓
- Failures are modelled via explicit error decision states and an
  explicit `state_error_callback` / `state_release_error_callback`
  publication state; no synthetic fault-injection knobs, no
  test-only control fields. ✓

### Context rules

- `action::context` holds only persistent actor-owned state (the
  slot pool and free stack). ✓
- No dispatch-local data is mirrored into context — request
  pointers, status codes, current handle, and attempt results all
  live in the per-dispatch `*_attempt_status` carriers attached to
  the internal runtime event. ✓
- Slot allocation is committed to context (because slots survive
  across dispatches by design — they hold the actor-owned mapped
  resource that the next `release_mapping` dispatch will unmap),
  but no per-dispatch *transient* state lives in context. ✓

### Naming

- States use `state_*`. Effects use `effect_*`. Guards live in the
  `guard::` namespace following the io/loader sibling-component
  convention. Constants use `k_` snake_case. New state names follow
  the `_decision` and `_callback` suffixes already established in
  Phases 204 and 205. ✓

### Domain and platform isolation

- The boundary-source test
  (`io mmap boundary keeps platform calls inside actions.cpp`)
  verifies that none of `::mmap(`, `munmap(`, `MapViewOfFile`,
  `CreateFileMapping`, `UnmapViewOfFile`, `<sys/mman.h>`,
  `<windows.h>`, `<fcntl.h>` appear in `actions.hpp`, `detail.hpp`,
  `sm.hpp`, `guards.hpp`, `events.hpp`, or `context.hpp`. ✓
- All platform headers are included only inside `actions.cpp`
  behind `#if defined(_WIN32)` selection. ✓
- Single platform-selection knob:
  `EMEL_IO_MMAP_PLATFORM_SUPPORTED` consumed only by
  `platform_mmap_supported` / `platform_mmap_unsupported` guards. ✓
- No leakage into `model/loader`, `model/tensor`, benchmark,
  paritychecker, embedded probe, or `src/emel/machines.hpp`. ✓

## Behavioural Bug Scan

Walked every reachable success and failure path; verified the slot
and OS-resource accounting for each:

- **Success path**: reserve slot → open → map → commit (writes base /
  mapped_bytes / os_resource into the slot, sets in_use=true via
  reservation) → publish done → ready. Slot remains in_use until
  release.
- **file_open failure**: reserve → open fails → release reserved
  slot, set in_use=false, push back onto free_stack, mark
  `file_open_failed`. The fd was never opened (open returned -1),
  so no os_resource leak.
- **mapping failure**: reserve → open → map fails → close the
  previously-opened os_resource → release reserved slot, mark
  `mapping_failed`. fd is closed; slot is freed.
- **resource exhaustion**: reserve guarded by
  `slot_capacity_available` (free_count > 0). When pool is drained,
  `slot_pool_exhausted` routes to error decision; no slot is
  consumed; no fd is opened. ✓
- **release happy path**: handle_in_range → slot_in_use →
  attempt_unmap (which snapshots base/bytes/os_resource from the
  slot) → unmap succeeds → release_slot_after_unmap (clears slot
  fields, pushes onto free_stack, sets ok=true) → publish done.
  After return, `slot.in_use == false` and `free_count` is
  incremented by 1. The next `map_tensor` dispatch reuses the
  released slot index (verified by the LIFO reuse test).
- **release out-of-range**: handle >= k_max_mappings → invalid
  handle. ctx.slots[handle] is never indexed because that index
  predicate is gated behind handle_in_range. ✓
- **release double**: first release marks slot in_use=false. Second
  release sees handle_in_range=true but slot_in_use=false → routes
  to invalid_handle. No double-unmap on the OS. ✓
- **release while slot was never committed (e.g., a fabricated
  handle)**: same as double-release — slot.in_use is false → routes
  to invalid_handle. ✓
- **unmap failure**: attempt_unmap reports unmap_ok=false →
  effect_mark_unmap_failed_and_release_slot still releases the slot
  bookkeeping (the actor never leaks a slot) and surfaces
  `error::unmap_failed`. The OS resources may have leaked at the
  syscall level, but that is reported deterministically and is the
  best the actor can do without retry semantics.

Edge-case math:

- `effect_reserve_top_free_slot_then_attempt_open` decrements
  `free_count` then indexes `free_stack[free_count]`. Reachable only
  after the `slot_capacity_available` guard accepted
  `free_count > 0`, so the indexed slot is well-defined. ✓
- The `layout_supported` predicate from Phase 205 still guards size
  against address-space wraparound before the platform attempt. ✓
- `effect_release_reserved_slot_on_open_failure` and
  `effect_close_open_resource_and_release_slot_on_mapping_failure`
  push the slot back via
  `ctx.free_stack[ctx.free_count] = ev.status.reserved_slot;
  ctx.free_count += 1u;`. Reachable only after a prior reservation
  decremented free_count, so the push slot index
  (`free_count` post-pop) is the same index the pop returned — pure
  LIFO. ✓

## Public Event Surface Contract Review

- `event::map_tensor_request::file_path` (`std::string_view`) MUST
  be backed by null-terminated storage that remains alive for the
  duration of dispatch, because `actions.cpp` calls
  `file_path.data()` for `open`/`CreateFileA`. The contract is
  documented in `206-CONTEXT.md`,
  `206-01-SUMMARY.md`, and `206-VERIFICATION.md`. Tests exercise it
  via `std::string` storage whose `c_str()` is guaranteed
  null-terminated. Phase 207 (model/tensor wiring) must respect this.
- `events::map_tensor_done.handle` is an opaque `uint32_t` slot
  index. Callers must treat it as opaque and pass it back via
  `event::release_mapping`. Out-of-range handles return
  `error::invalid_request`.
- `event::release_mapping.handle` defaults to
  `k_invalid_mapping_handle` so an explicit handle is required at
  construction.
- `events::map_tensor_done` carries `const event::map_tensor &request`
  and `events::map_tensor_error` carries the same back-pointer for
  same-RTC correlation. Phase 207 must not store either past
  dispatch return.

## Portability Review

- `actions.cpp` is the single TU with platform-specific code.
  POSIX includes `<fcntl.h>`, `<sys/mman.h>`, `<sys/stat.h>`,
  `<unistd.h>`. Windows includes `<windows.h>` with
  `WIN32_LEAN_AND_MEAN`.
- `intptr_t` is used for `os_resource` so the same field stores a
  POSIX `int fd` or a Windows `HANDLE` (which is itself a
  `void*`-sized value). `reinterpret_cast<HANDLE>` and
  `reinterpret_cast<intptr_t>` round-trip safely on both targets.
- `EMEL_IO_MMAP_PLATFORM_SUPPORTED` defaults to `1` on
  `__APPLE__`, `__linux__`, `__unix__`, or `_WIN32`. Other targets
  (e.g., bare metal) leave it at `0` and fail closed at
  `state_platform_decision`.
- Tests are POSIX-only in practice (filesystem temp paths under
  `/tmp`, directory mapping at `/`). Cross-platform CI for Windows
  would benefit from Windows-specific test cases; recorded as an
  observation for Phase 209.

## Test Coverage Scan

`tests/io/mmap/lifecycle_tests.cpp` exercises (18 cases / 672
assertions):

- Canonical aliases.
- All Phase 205 validation outcomes via the public event surface.
- New: empty file_path → invalid_request.
- New: file_open_failed via missing path.
- New: mapping_failed via directory fd (with a permissive assertion
  for platforms that reject the directory `open` up front).
- New: deterministic descriptor on success with byte-level content
  verification.
- New: success without `on_done` (record path).
- New: release happy path with LIFO slot-reuse verification.
- New: release out-of-range handle.
- New: release double-release.
- New: release without callbacks.
- New: resource_exhausted by filling 256 slots and dispatching one
  more.
- Boundary-source check that platform identifiers appear in
  `actions.cpp` only.

Coverage gaps that are correctly deferred:

- `effect_mark_unsupported_platform` body — dead code on
  `EMEL_IO_MMAP_PLATFORM_SUPPORTED == 1` builds (all currently
  supported hosts). Phase 209 may add a compile-toggled coverage
  build.
- `effect_mark_unmap_failed_and_release_slot` body — reachable only
  when `munmap` / `UnmapViewOfFile` reports failure; cannot be
  deterministically triggered without sabotaging the platform
  helper. Phase 209 owns this.
- `effect_on_unexpected` internal_error branch for the
  release_mapping_runtime sentinel — owned by Phase 209.
- A handful of release-side callback-presence guard branches
  (`guards.hpp:111,113,219,221`) — owned by Phase 209.

## Phase 207 Deferral Verification

Phase 206 correctly does NOT contain:

- Any `model/tensor` change. `event::map_tensor_request` is the
  io/mmap-owned event surface; how `model/tensor` populates
  `file_path` is Phase 207's job.
- Any `model/loader` change.
- Any benchmark, paritychecker, embedded probe, or
  `src/emel/machines.hpp` modification. The `emel::IoMmap` alias
  still resolves through the existing `using` declaration without
  Phase 206 needing to touch it.
- Any cooperative async, register-once-then-map-many, staged
  read/copy, or device-strategy hooks.

These deferrals match Phase 206 scope and the v1.24 ROADMAP.

## Informational Observations (non-blocking)

1. **POSIX directory-mapping test relies on platform behaviour.**
   `io mmap surfaces mapping_failed when mmap call fails` opens `/`
   and expects `mmap` to fail. On Linux and macOS this currently
   fails as `EACCES` / `ENODEV`. The test's assertion accepts
   either `mapping_failed` or `file_open_failed` to handle hosts
   that reject directory opens up front. This is fine for current
   POSIX targets; Phase 209 may want a more deterministic
   sabotage harness for Windows coverage.

2. **fd-limit considerations for the resource_exhausted test.**
   The test maps the same file 256 times to drain the slot pool,
   which consumes 256 fds simultaneously before releasing them.
   On hosts whose `ulimit -n` is below ~270 the test would fail
   prematurely with `file_open_failed` instead of
   `resource_exhausted`. macOS and Linux defaults are typically
   high enough; CI runners with tight rlimits should be checked.
   A future tightening could compile-override
   `EMEL_IO_MMAP_MAX_MAPPINGS` to a small value for the test
   binary so the test costs only a few fds.

3. **slot fields on the rollback paths.** When a slot is released
   on the open-failure or mapping-failure path, only `in_use` is
   reset; `base`, `mapped_bytes`, `os_resource`,
   `file_offset`, `requested_bytes` keep their stale values until
   the next `effect_commit_mapping` or `effect_release_slot_*`
   overwrites them. This is correct because no consumer reads stale
   slot fields before the next commit, but a defensive zeroing
   would make the slot record easier to inspect during debugging.
   Not a Phase 206 issue.

4. **Outcome event back-pointers.** Both
   `events::map_tensor_done` and `events::map_tensor_error` (and the
   release variants) carry a `const event::*&` back-pointer for
   same-RTC correlation. Phase 207 wiring through `model/tensor`
   must not store either outcome event past the synchronous callback
   return.

5. **Phase 204 transitional bench-regression override.** Still
   carry-forward; not consumed by Phase 206 (no benchmark-affecting
   changed files this phase). Phase 210 owes the cleanup.

## Status

Clean. Phase 207 may proceed without rework of any Phase 206 io/mmap
artifact.
