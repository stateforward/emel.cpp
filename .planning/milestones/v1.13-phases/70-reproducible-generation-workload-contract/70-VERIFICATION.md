---
phase: 70-reproducible-generation-workload-contract
status: complete
verified: 2026-04-20T23:25:00Z
---

# Phase 70 Verification

## Commands

- `cmake --build build/bench_tools_ninja --parallel --target bench_runner bench_runner_tests`
- `./build/bench_tools_ninja/bench_runner_tests --test-case="bench_runner generation compare keeps maintained Qwen and Liquid fixtures"`
- `./build/bench_tools_ninja/bench_runner_tests --test-case="bench_runner generation jsonl emits manifest-driven workload metadata and explicit comparability"`
- `ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests`
- `./scripts/quality_gates.sh`

## Results

- The maintained bench build succeeded with the new manifest loader and generation compare record
  fields.
- The maintained compare doctest still proved the Qwen3 and LFM2 compare fixtures remained on the
  compare path.
- The JSONL doctest proved manifest-driven output provenance:
  - `workload_id`
  - `workload_manifest_path`
  - `prompt_fixture_id`
  - `prompt_fixture_path`
  - `formatter_mode`
  - explicit `comparable` truth
- `ctest -R bench_runner_tests` passed in `406.06 sec`.
- `./scripts/quality_gates.sh` passed end to end:
  - coverage lines: `90.4%`
  - coverage branches: `55.0%`
  - paritychecker: passed
  - fuzz smoke: passed
  - benchmark snapshot tail: completed under existing warning-tolerant gate policy
  - docs generation: passed

## Requirements

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| `WRK-01` | `70-01` | Operator can run compare workloads from explicit manifests that pin model identity, prompt fixture identity, formatter mode, seed, and sampling parameters. | passed | `tools/bench/generation_workloads/` and `tools/bench/generation_prompts/` became the checked-in workload source of truth; JSONL doctests verified manifest-driven fields. |
| `WRK-02` | `70-01` | Compare artifacts preserve enough workload metadata to reproduce a run on the same engine and explain mismatches across engines. | passed | `generation_compare/v1` records include workload manifest path, prompt fixture, formatter, comparability, and timing/output fields; Phase 75 expanded summary metadata for drift review. |
| `WRK-03` | `70-01` | Workflow rejects or marks non-comparable runs when formatter, tokenization, or sampling contracts materially diverge across lanes. | passed | Phase 70 introduced `comparison_mode` and `comparable`; Phase 75 added verdict checks for formatter contract, sampling, stop, seed, and token-budget mismatches. |
