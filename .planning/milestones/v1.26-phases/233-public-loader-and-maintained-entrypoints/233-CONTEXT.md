# Phase 233 Context — Public Loader and Maintained Entrypoints

## Milestone slice

**ROADMAP:** `PUB-01`–`PUB-05` (strategies observable through **public** contracts on loader,
benchmarks, parity, embedded-size probes; no unconstrained duplicate POSIX staged-read shims in tools).

## Source-backed facts verified for planning (this closeout assist)

All paths under `src/emel/` or `tools/` unless noted.

### `emel::io::loader` strategy enum

`emel::io::loader::event::strategy_kind` (`src/emel/io/loader/events.hpp`) enumerates
`external_buffer = 3u` and `staged_read = 4u` (with `none`, `mapped_file`, `read_copy` below them).

### Loader context carries staged actor pointer

`emel::io::loader::action::context` (`src/emel/io/loader/context.hpp`) holds
`emel::io::staged_read::sm *io_staged_read` alongside `io_read`.

### Single vs batch staged dispatch (public events only)

- **Single-tensor route:** `sm.hpp` transitions into `state_staged_read_dispatch_decision` with
  `action::effect_dispatch_staged_read_tensor`, which constructs
  `emel::io::staged_read::event::staged_window` and calls `io_staged_read->process_event(...)`
  (`src/emel/io/loader/actions.hpp`, `sm.hpp`).
- **Batch route:** batch transitions use `effect_dispatch_staged_read_tensor_batch`, which builds
  `emel::io::staged_read::event::staged_window_batch` and dispatches via `process_event(batch)`
  (same files; batch contract is `std::span`-based in `events.hpp`).

### Loader does not include staged_read `detail` / non-public orchestration headers

`src/emel/io/loader/actions.hpp` includes **`emel/io/staged_read/errors.hpp`**, **`events.hpp`**,
and **`sm.hpp`** only (no `detail.hpp`, no `guards.hpp` from `staged_read`).
`src/emel/io/loader/guards.hpp` includes **`staged_read/errors.hpp`** only.

### `tools/bench` strategy naming

`tools/bench/model_load_strategy.hpp` maps env `staged` / `staged_read` to
`strategy_kind::staged_read` and names it `staged_read` in `model_load_io_strategy_name`.

### `model/loader` boundary (PUB-01)

`src/emel/model/loader/actions.hpp` includes **`emel/io/loader/sm.hpp`** (public loader machine);
`src/emel/model/loader/events.hpp` includes **`emel/io/loader/events.hpp`**. No direct
`emel/io/staged_read/**` includes observed in `model/loader` (integration stays behind `io::loader`).

## Out of scope for this context file

- Claiming **paritychecker** or **probe** entrypoints meet `PUB-03` / `PUB-04` (needs lane-specific
  evidence in verification).
- Claiming **`scripts/quality_gates.sh` pass** for Phase 233 (not run in this assist unless recorded
  elsewhere).
