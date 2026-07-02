---
gsd_state_version: 1.0
milestone: v1.27
milestone_name: Ryzen AVX2/FMA Kernel Support
status: ready_for_next_milestone
stopped_at: "v1.27 shipped and archived; next step is new milestone definition"
last_updated: "2026-06-25T14:35:28.048Z"
last_activity: 2026-06-25
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 6
  completed_plans: 6
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-25)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.

**Current focus:** Define the next milestone after shipping native x86_64
AVX2/FMA support for this AMD Ryzen 9 5950X host.

## Current Position

Milestone: v1.27 Ryzen AVX2/FMA Kernel Support
Status: v1.27 shipped and archived. Phases 239-244 verified and the
source-backed milestone audit passed. The repaired `kernel_x86_64` benchmark
suite includes counter-checked optimized flash and q2/q3/q6 rows.
Phase: ready for next milestone definition.
Last activity: 2026-06-25 — `snapshots/bench/benchmarks.txt` plus the
maintained LFM2 `10`, `100`, and `1000` token generation baselines were
updated, the source-backed `XBN-01` benchmark gap was repaired, the x86_64
unary SML rule debt was removed, focused validation and the changed-file scoped
quality gate passed, the milestone audit passed, and the milestone was archived.

Progress: [##########] 100%

**Host feature scope:**

- CPU: AMD Ryzen 9 5950X 16-Core Processor.
- Supported target features: x86_64 AVX2, FMA, and F16C conversion.
- Explicit no-claim features: AVX-512, AVX-VNNI, AMX, BF16, native FP16, and GPU
  acceleration.

**Next implementation step:** define the next milestone with `$gsd-new-milestone`.

**Closeout gate:** complete.

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

- Quantized kernel parity requires the same effective operand class as the
  reference path; whole-tensor dequantize-to-f32 hot-path substitution requires
  explicit user approval and is not part of v1.27.

- Benchmark and parity claims must be source-backed by the maintained runtime
  path, not only planning artifacts or tool-local scaffolds.

### Carry-Forward Backlog

- 2026-04-02 - Move eager quant prepack into generator initializer
  (`.planning/todos/backlog/2026-04-02-move-eager-quant-prepack-into-generator-initializer.md`)

- 2026-04-02 - Reuse q8 RHS across LFM2.5 prefill matmuls
  (`.planning/todos/backlog/2026-04-02-reuse-q8-rhs-across-lfm2-5-prefill-matmuls.md`)

- 2026-04-02 - Optimize LFM2.5 q4 prefill kernel
  (`.planning/todos/backlog/2026-04-02-optimize-lfm2-5-q4-prefill-kernel.md`)

- 2026-04-02 - Optimize LFM2.5 q6 prefill kernel
  (`.planning/todos/backlog/2026-04-02-optimize-lfm2-5-q6-prefill-kernel.md`)

These pre-existing LFM2.5 performance backlog items are outside the v1.27 Ryzen
AVX2/FMA support contract and are not milestone close blockers.

### Blockers/Concerns

- `ESG-02B` from v1.26 remains outside v1.27 processor scope until a file-backed
  staged-read source path is separately approved.

- v1.27 must not present AVX-512/VNNI/AMX/BF16/native-FP16 claims for this host.

### Prior milestone notes

`v1.26 I/O Staged Read Loading Strategy` completed on 2026-05-08. Its final
audit passed after Phase 237 repaired direct tensor staged-load nonzero-offset
behavior and Phase 238 reconciled artifact/reporting truth. Active v1.26
evidence is archived under `.planning/milestones/v1.26-*`.

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

Last session: 2026-06-25 (v1.27 closeout)
Stopped at: v1.27 shipped and archived; ready for the next milestone.
Resume file: None
