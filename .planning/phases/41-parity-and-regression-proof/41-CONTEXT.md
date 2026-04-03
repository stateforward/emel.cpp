---
phase: 41
slug: parity-and-regression-proof
created: 2026-04-02
status: ready
---

# Phase 41 Context

## Phase Boundary

Prove the maintained `tests/models/Bonsai-1.7B.gguf` slice against the pinned Prism reference lane
on the same formatter contract and add regression coverage that protects Bonsai plus the prior
maintained Qwen and Liquid anchors from fixture, formatter, architecture, or quant-path drift.

## Implementation Decisions

### Current Parity Surface
- `tools/paritychecker --generation` currently accepts only
  `k_supported_generation_fixtures`, which still excludes Bonsai.
- The maintained generation path compares EMEL output against append-only baseline files under
  `snapshots/parity/`; it does not currently compare against a live reference generation result.
- The parity harness already loads the pinned reference backend and contains
  `run_reference_generate(...)`, so live Prism generation proof is available in-process but not
  wired into the maintained `--generation` contract.

### Explicit Grey Area
- The roadmap says Phase 41 must prove against a pinned `PrismML-Eng/llama.cpp` reference lane.
- The existing maintained-generation contract is append-only baselines, which is narrower than live
  reference comparison unless the baseline authoring path is itself treated as the audited Prism
  proof.
- Adding Bonsai to the maintained generation set also requires new append-only baseline files under
  `snapshots/parity/`, which is a snapshot update and therefore needs explicit user consent.
- A single global `reference_repo.txt` plus `reference_ref.txt` pair is not sufficient for the
  maintained fixture set. Using the Prism pin globally breaks live Qwen parity, so the reference
  identity has to be modeled per fixture as `(model, engine, commit?)`, with the commit pin
  optional for fixtures that follow the engine default branch.
- The prior maintained anchors did not ship a live-reference generation contract. Their truthful
  regression protection surface is still the existing append-only baseline contract, while Bonsai
  is the new live-reference proof target for this phase.

### Pinned Reference Truth
- `tools/paritychecker/reference_ref.txt` currently pins the Prism reference lane to
  `ecbcb7ea9d3303097519723b264a8b5f1e977028`.
- Repo-visible parity evidence should surface that exact ref when Bonsai proof runs.

## Existing Code Insights

### Reusable Assets
- `tools/paritychecker/parity_runner.cpp` already exposes:
  - maintained fixture gating via `find_generation_fixture(...)`
  - live reference generation via `run_reference_generate(...)`
  - formatter contract publication via `print_generation_formatter_contract(...)`
  - reference identity publication via `reference_impl: source=... ref=...`
- `tools/paritychecker/paritychecker_tests.cpp` and `tools/bench/bench_runner_tests.cpp` already
  enumerate maintained generation fixtures from `k_supported_generation_fixtures`, so promoting
  Bonsai there will automatically widen regression coverage.
- `tools/generation_fixture_registry.hpp` already contains full maintained Bonsai metadata; only
  `generation_supported` and the supported-fixtures array still keep it out of the maintained
  parity/bench path.

### Integration Points
- `tools/generation_fixture_registry.hpp` must promote Bonsai into the maintained supported
  generation set once Phase 41 decides the proof contract.
- `tools/paritychecker/parity_runner.cpp` must either:
  - compare Bonsai against a live Prism result and optionally author audited baselines, or
  - explicitly document and enforce that maintained baselines are authored from the pinned Prism
    lane and remain the comparison contract.
- `scripts/paritychecker.sh` must orchestrate the reference build per fixture tuple instead of
  assuming one global repository/ref pair for every maintained model.
- `tools/paritychecker/paritychecker_tests.cpp` must gain Bonsai-focused generation proof and
  regression assertions without weakening the maintained fixture-path restrictions.
