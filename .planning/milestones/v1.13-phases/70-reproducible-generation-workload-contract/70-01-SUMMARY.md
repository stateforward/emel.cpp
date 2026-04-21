---
phase: 70-reproducible-generation-workload-contract
plan: 01
status: complete
completed: 2026-04-20
requirements-completed:
  - WRK-01
  - WRK-02
  - WRK-03
---

# Phase 70 Summary

## Outcome

Phase 70 is complete. Maintained generation workloads are now driven by checked-in prompt and
workload manifests, and `generation_compare/v1` records carry enough provenance to replay the
same run and distinguish parity workloads from single-lane-only workloads truthfully.

## Delivered

- Added `tools/bench/generation_prompts/` and `tools/bench/generation_workloads/` as the
  checked-in source of truth for maintained prompt identity, model fixture identity, formatter
  mode, sampling, stop conditions, seed, token budgets, and comparability.
- Added `tools/bench/generation_workload_manifest.hpp` and rewired `generation_bench.cpp` to load
  maintained workload metadata from manifests before timed dispatch.
- Extended `generation_compare/v1` with workload-manifest provenance, prompt-fixture provenance,
  formatter mode, and explicit `comparable` truth.
- Marked the EMEL-only Gemma4 workloads as `single_lane` instead of letting them look like parity
  workloads.

## Contract Truth

- Prompt identity now comes from `tools/bench/generation_prompts/single_user_hello.json`.
- Maintained comparable generation workloads now come from the checked-in Qwen3 and LFM2 workload
  manifests under `tools/bench/generation_workloads/`.
- The maintained Gemma4 workloads stay available for EMEL-only bench output, but they are now
  explicitly labeled:
  - `comparison_mode=single_lane`
  - `comparable=false`
  - `note=reference_lane_unavailable_for_maintained_compare_surface`

## Verification Result

- `bench_runner` build and focused generation doctests passed.
- `ctest -R bench_runner_tests` passed with the manifest-driven workload contract enabled.
- `./scripts/quality_gates.sh` passed end to end after the workload-manifest refactor.
- The milestone is now ready for Phase `71`, which can move the first maintained non-EMEL
  reference backend behind a pluggable generation compare surface.
