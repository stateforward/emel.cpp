---
phase: 208-public-runtime-and-evidence-surfaces
status: passed
requirements:
  - TIO-03
  - VAL-04
created: 2026-05-04T22:15:00Z
last_updated: 2026-05-04T22:15:00Z
backfilled_by: 211-phase-verification-artifact-backfill
---

# Phase 208 Verification

## Source-Backed Requirement Check

This file backfills the per-phase verification artifact required by the milestone
audit's 3-source cross-reference gate. The Requirement Status content was originally
inlined in `208-VALIDATION.md`; Phase 211 promotes it here without changing runtime
code, tests, snapshots, or maintained gate evidence. All source-backed evidence below
was independently re-checked against the live repository at audit time.

### TIO-03 — `model/loader`, maintained benchmark/parity/probe lanes select or report mmap-backed loading only through public runtime surfaces

| Maintained Lane | Source Evidence | Status |
|------------------|------------------|--------|
| `model/loader` actions | `src/emel/model/loader/actions.hpp:166` initializes `ev.ctx.used_mmap = false` and `:381` propagates that value into `events::load_done.used_mmap` (`grep -n "used_mmap" src/emel/model/loader/actions.hpp` returns exactly two matches). No `mmap_resident`, `tensor_state`, `capture_tensor_state`, or `lifecycle::*` reference appears in loader actions. | passed |
| Benchmark lane | `tools/bench/generation_bench.cpp:753` constructs `emel::model::tensor::event::capture_tensor_state` and dispatches via `process_event(...)`; lifecycle classification compares to `emel::model::tensor::event::lifecycle::mmap_resident`. No internal include from `model/tensor/{actions,detail,guards}` or `io/mmap/{actions,detail,guards}`. | passed |
| Paritychecker lane | `tools/paritychecker/parity_engines.cpp:1312` mirrors the same public `capture_tensor_state` event pattern. No actor-internal includes. | passed |
| Embedded probe lane | `tools/embedded_size/emel_probe/main.cpp:487` mirrors the same public pattern. No actor-internal includes. | passed |
| Tool reach-through scan | `grep -rn "model/tensor/actions\|model/tensor/detail\|model/tensor/guards\|io/mmap/actions\|io/mmap/detail\|io/mmap/guards" tools/` returns 0 matches. | passed |

### VAL-04 — Maintained benchmark/parity evidence reports mmap usage only when the EMEL lane actually runs the mmap-backed runtime path

| Check | Source Evidence | Status |
|-------|------------------|--------|
| Loader does not infer mmap from tensor residency | `src/emel/model/loader/actions.hpp:166` hard-codes `used_mmap = false`; loader never sets it `true`. | passed |
| Tools report mmap usage via public events | All three maintained tool lanes read mmap residency through `event::capture_tensor_state` only; no `mmap_resident` derivation in the maintained tool code paths. | passed |
| No fake/derived mmap claim | Lifecycle classification in `tools/bench/generation_bench.cpp:760`, `tools/paritychecker/parity_engines.cpp:1319`, `tools/embedded_size/emel_probe/main.cpp:494` compares against the public `lifecycle::mmap_resident` enumerator only. | passed |

## Result

Both TIO-03 and VAL-04 are source-backed verified. No code, test, or snapshot
contradiction. Phase 211 closes the artifact-format gap that prevented the milestone
audit's 3-source cross-reference from passing for these requirements.
