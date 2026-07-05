---
gsd_state_version: 1.0
milestone: v1.28
milestone_name: Memory-Owned KV Block Addressing Cutover
status: active
stopped_at: "Phases 245-249 complete + KVP-01; v1.28 closeout blocked on bench-lane repair (chip task_48a05fc3)"
last_updated: "2026-07-05T00:00:00.000Z"
last_activity: 2026-07-05
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 6
  completed_plans: 5
  percent: 83
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-25)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.

**Current focus:** Cut physical KV-cache addressing over to the memory domain's
block map on the maintained generation path (v1.28), with bit-exact single-sequence
parity and component-level multi-sequence proof.

## Current Position

Milestone: v1.28 Memory-Owned KV Block Addressing Cutover
Status: Phases 245-249 complete and committed with per-phase committed-state gate
evidence (coverage + parity lanes pass; bench lane blocked by pre-existing main
breakage). Phase 250: KVP-01 multi-sequence component proof landed; KVE-01/KVD-01
closeout blocked on the bench-lane repair (reference ggml SIGBUS + missing PR #89
baselines, chip task_48a05fc3). Full suite 14/14 (1816 cases) on the unsharded
zig build.
Phase: 250 closeout pending upstream bench repair and consent decisions.
Last activity: 2026-07-05 — PR #94 takeover fixed review-loop findings around
KV reuse order, graph execute KV-addressing validation, identity addressing,
snapshot-coverage guard cost, and unapproved lint-snapshot churn. See phase
VALIDATION docs under .planning/phases/24{5,6,7,50}-*/ for earlier gate evidence
and dispositions. A final all-files union gate run timed out at 30 minutes under
host contention (a second agent session was running); per-phase committed-state
evidence stands.

Progress: [########--] 83%

**Audit findings driving this milestone:**

- `memory::hybrid` is dispatched for real on the generate path
  (`allocate_sequence`/`free_sequence`/`capture_view`), but physical K/V bytes live
  in generator-backend vectors addressed linearly (`kv_cache_tokens = position + 1`).
- The captured `memory::view::snapshot` is threaded into graph/processor events and
  never dereferenced in `src/` (test-only consumption).
- `sequence_recurrent_slot` exists in the view contract but is unused; shortconv
  state uses a flat layer-only offset.
- Cache layouts: linear `[layer][position][head][dim]` and flash
  `[layer][head][position][dim]`, both written per token
  (`store_attention_kv_cache`, generator `detail.hpp`).
- Flash kernels consume strided 3D views that assume position-contiguity per head —
  flash eligibility under block mapping must be an explicit guard.

**Next implementation step:** no further v1.28 implementation is queued in this
PR. KVE-01/KVD-01 closeout waits on bench-lane repair and explicit snapshot
baseline consent.

**Closeout gate:** open (5/6 phases complete). KVP-01 landed; KVE-01/KVD-01 stay
blocked by the bench lane.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Carry-forward architectural constraints:

- Runtime behavior selection remains explicit guards and transitions
  (`AGENTS.md` / `docs/rules/sml.rules.md`).

- Kernel arithmetic, lowering, packing, quant/dequant, and backend-specific
  numeric work stays in the owning kernel layer.

- The EMEL lane stays repo-owned and separate from llama.cpp/ggml reference
  runtime state; reference linkage is comparison-only in tools.

- Benchmark and parity claims must be source-backed by the maintained runtime
  path, not only planning artifacts or tool-local scaffolds.

- v1.28-specific: addressing math (logical position -> physical block offset) is
  data-plane detail work; route selection (flash vs span-walk, error routing) is
  guards/transitions only. Public generation API stays single-sequence this
  milestone. Dual linear+flash K/V storage is explicitly out of scope to change.

### Carry-Forward Backlog

- 2026-04-02 - Move eager quant prepack into generator initializer
  (`.planning/todos/backlog/2026-04-02-move-eager-quant-prepack-into-generator-initializer.md`)

- 2026-04-02 - Reuse q8 RHS across LFM2.5 prefill matmuls
  (`.planning/todos/backlog/2026-04-02-reuse-q8-rhs-across-lfm2-5-prefill-matmuls.md`)

- 2026-04-02 - Optimize LFM2.5 q4 prefill kernel
  (`.planning/todos/backlog/2026-04-02-optimize-lfm2-5-q4-prefill-kernel.md`)

- 2026-04-02 - Optimize LFM2.5 q6 prefill kernel
  (`.planning/todos/backlog/2026-04-02-optimize-lfm2-5-q6-prefill-kernel.md`)

These pre-existing LFM2.5 performance backlog items are outside the v1.28 contract
and are not milestone close blockers.

### Blockers/Concerns

- `ESG-02B` from v1.26 remains out of scope until a file-backed staged-read source
  path is separately approved.

- Open enhancement issues not in this milestone: #64 (co_sm cooperative async I/O
  strategy; superseded local plan recoverable at commit f23a16b2), #57
  (loader/parser/weight-loader C API parity).

- Dual linear+flash K/V storage costs 2x KV memory; noted, deliberately unchanged
  in v1.28 (performance-contract decision deferred to the user).

### Prior milestone notes

`v1.27 Ryzen AVX2/FMA Kernel Support` shipped 2026-06-25; archived under
`.planning/milestones/v1.27-*`. Post-ship merges on main: Mimi codec Moshi parity
(#90), aarch64 row-sliced lane guard fix (#92), x86_64 AVX2+FMA quantized matmul
parity (#89).

## Historical Carry-Forward Items

Items acknowledged at v1.25 milestone close on 2026-05-06 (unchanged):

| Category | Item | Status |
|----------|------|--------|
| quick_task | 260401-ejm-add-non-blocking-benchmark-binary-size-c | complete |
| todo | 2026-04-02-move-eager-quant-prepack-into-generator-initializer.md | backlog |
| todo | 2026-04-02-optimize-lfm2-5-q4-prefill-kernel.md | backlog |
| todo | 2026-04-02-optimize-lfm2-5-q6-prefill-kernel.md | backlog |
| todo | 2026-04-02-reuse-q8-rhs-across-lfm2-5-prefill-matmuls.md | backlog |

## Session Continuity

Last session: 2026-07-05 (PR #94 takeover)
Stopped at: Phases 245-249 complete plus KVP-01; KVE-01/KVD-01 intentionally
blocked on the bench-lane repair and snapshot consent.
Resume file: None
