---
phase: 71-maintained-reference-backend-integration
status: complete
verified: 2026-04-21T02:18:27Z
---

# Phase 71 Verification

## Commands

- `cmake --build build/bench_tools_ninja --parallel --target generation_compare_tests`
- `./build/bench_tools_ninja/generation_compare_tests`
- `ctest --test-dir build/bench_tools_ninja --output-on-failure -R generation_compare_tests`

## Results

- The generation compare driver produced `generation_compare_summary/v1` summaries from canonical
  `generation_compare/v1` records.
- The missing backend executable path produced an explicit reference-lane error record rather than
  corrupting EMEL JSONL output.
- Wrapper smoke coverage proved `--skip-emel-build` and reference `--run-only` modes stay confined
  to tooling surfaces.

## Requirements

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| `REF-01` | `71-01` | At least one maintained non-EMEL generative backend can run through the shared compare contract on the canonical generation slice. | passed | `tools/bench/reference_backends/llama_cpp_generation.json` and `scripts/bench_generation_reference_llama_cpp.sh` define the maintained llama.cpp generation backend; Phase 73 and 75 wrapper regressions run it end to end. |
| `REF-02` | `71-01` | Backend-specific setup stays confined to the reference lane and does not leak into `src/` runtime code or the EMEL compute path. | passed | Backend build/run setup lives in the reference wrapper and backend manifest; Phase 74 verified EMEL JSONL no longer prepares reference fixtures. |
| `REF-03` | `71-01` | Backend failures surface explicit, reproducible errors without corrupting EMEL results or compare summaries. | passed | `generation_compare_tests` covers missing backend executable behavior and verifies an explicit reference-lane error record and failed summary. |
