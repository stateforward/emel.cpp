# Phase 145 Context: Text Generator Native Quantized Route Evidence Closure

## Source-backed trigger

The v1.17 milestone audit found that TEXTGEN-04 and TEXTGEN-07 remain blocked by
dispatch-time behavior selection in the maintained text generator native quantized route.

The concrete blocker is in `src/emel/text/generator/detail.hpp`:

- `matmul_vector_native_quantized(...)` probes `packed_q8_0_input_path_supported(...)` and
  `q8_input_path_supported(...)` from inside action-called detail code.
- The existing source regression checks a different `matmul_vector(...)` body and misses this
  helper.
- `phase_lifecycle(...)` uses a runtime-indexed lifecycle manifest array from action-called
  detail code.

## Constraints

- Keep Stateforward.SML orchestration as the behavior source of truth.
- Do not introduce dispatch-time queues, deferred work, or hidden route selection in actions or
  detail helpers.
- Preserve native quantized behavior and optimized kernel evidence where the prepared tensor
  layouts already provide it.
- Add a failing regression before the fix.
- Run focused generator tests, domain boundary checks, parity evidence, and a scoped quality gate.

## User authorization

The user explicitly authorized changing models, benchmarks, and snapshots if required. This phase
does not plan to change model contracts or snapshots unless verification proves they are stale.
