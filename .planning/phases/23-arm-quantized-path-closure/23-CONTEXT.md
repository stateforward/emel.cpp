# Phase 23: ARM Quantized Path Closure - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 23 must close any remaining supported canonical ARM branches that still widen through
disallowed whole-row or operator-level dequantize-to-f32 substitution. If Phase 22's audit is
already proving zero disallowed fallback on supported canonical requests, this phase must make that
zero-gap truth explicit in the shipped runtime and proof surfaces instead of inventing artificial
work.

This phase stays inside the existing generator -> graph -> processor -> kernel chain. It must not
change Boost.SML actor structure, public C API boundaries, or maintained acceptance surfaces
without explicit approval.

</domain>

<decisions>
## Implementation Decisions

### Locked From Phase 22
- The canonical truth source for the maintained ARM slice is now
  `emel::model::llama::detail::build_quantized_path_audit(...)` over the shipped Llama
  `execution_view`.
- Supported canonical q2/q3/q6 matmul stages are published as `native_quantized`.
- Token embedding row copy and norm-vector stages are published as
  `approved_dense_f32_by_contract`.
- Unsupported quantized stage families publish explicit `no-claim` behavior instead of inheriting
  approved contract labels.

### Phase 23 Question
- Determine whether any supported canonical request still uses a disallowed whole-row or
  operator-level dequantize-to-f32 substitution in `src/`.
- If no supported disallowed gap remains, close `PATH-01` by hardening the shipped runtime and its
  proof surface around that zero-gap contract rather than by forcing a narrower milestone than the
  codebase currently needs.

### Guardrails
- Do not change Boost.SML transition tables or actor ownership without explicit user approval.
- Preserve the existing canonical ARM acceptance boundary in `tools/paritychecker` and
  `tools/bench`.
- Keep any closure work additive, deterministic, and grounded in the shipped runtime chain.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/ROADMAP.md` — Phase 23 goal, guardrails, and success criteria.
- `.planning/REQUIREMENTS.md` — `PATH-01` definition.
- `.planning/PROJECT.md` — current milestone intent and validated Phase 22 decisions.
- `.planning/phases/22-quantized-path-audit-and-contract/22-VERIFICATION.md` — current audit
  truth and proof surface.
- `AGENTS.md` and `docs/rules/sml.rules.md` — no machine-structure change without approval and RTC
  actor constraints.

</canonical_refs>

<code_context>
## Existing Code Insights

- `src/emel/generator/detail.hpp` is the shipped source of truth for token embedding row copy,
  norm-vector dequantization, and quantized matmul binding.
- `src/emel/kernel/detail.hpp` and `src/emel/kernel/aarch64/actions.hpp` define the supported
  `q*_K x q8_K` kernel contract and any fallback or rejection boundary.
- `src/emel/model/data.cpp` and `src/emel/model/llama/detail.hpp` now expose the Phase 22
  quantized-path audit helper that later proof phases can reuse.
- `tools/paritychecker/parity_runner.cpp` already publishes stage inventory and per-stage audit
  rows for the canonical maintained workload.

</code_context>

<specifics>
## Specific Ideas

- Prove or refute that any supported canonical ARM request still widens through disallowed
  dequantize-to-f32 substitution in shipped runtime code.
- If the gap is already zero, add the narrowest runtime-proof surface needed so later parity and
  regression phases can fail deterministically if that contract regresses.
- Keep closure work separate from broader proof expansion across `1/10/100/1000`; that is still
  Phase 24.

</specifics>

<deferred>
## Deferred Ideas

- Broader paritychecker regression hardening across all maintained decode lengths remains Phase 24.
- Benchmark attribution refresh remains Phase 25.

</deferred>

---
*Phase: 23-arm-quantized-path-closure*
*Context gathered: 2026-03-25*
