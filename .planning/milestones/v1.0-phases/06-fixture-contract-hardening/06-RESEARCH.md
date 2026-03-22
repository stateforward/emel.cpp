# Phase 06: Fixture Contract Hardening Research

**Phase:** 06
**Date:** 2026-03-08
**Requirement IDs:** HARN-02
**Source:** Milestone audit `v1.0-MILESTONE-AUDIT.md`

## Goal

Close the audit gap in `HARN-02` by making the Phase 1 generation fixture contract fully
unambiguous at the paritychecker boundary and proving that contract through automated subprocess
coverage.

## What Exists Today

- `tools/paritychecker/parity_runner.cpp` defines
  `k_generation_fixture_name = "Llama-68M-Chat-v1-Q2_K.gguf"`.
- `is_expected_generation_fixture(...)` currently accepts any model path whose basename matches the
  pinned fixture name.
- `run_generation_harness_contract(...)` rejects non-matching basenames but does not require the
  canonical `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` location.
- `tools/paritychecker/parity_main.cpp` still prints stale help text saying generation "reserves"
  the contract even though the path is fully wired.
- `tools/paritychecker/paritychecker_tests.cpp` already has a canonical success case via
  `models_dir() / "Llama-68M-Chat-v1-Q2_K.gguf"` and one failure case for a missing model path, but
  no regression that proves basename-only impostors are rejected.

## Audit Gap

The milestone audit classified `HARN-02` as partial because the implementation accepts any file
whose basename is `Llama-68M-Chat-v1-Q2_K.gguf`. That means the requirement wording
"without ambiguous fixture selection" is stronger than the runtime contract.

## Narrowest Correct Fix

Keep the fix confined to `tools/paritychecker/` and the existing test surface:

1. Define one canonical generation fixture path for paritychecker rather than a basename-only rule.
2. Normalize both the expected path and the provided `--model` path before comparison so relative
   vs absolute spelling does not make the contract flaky.
3. Update the help/usage text and runtime diagnostic strings so they describe the same canonical
   rule the code enforces.
4. Add a subprocess regression that creates or copies a same-basename file outside the canonical
   `tests/models/` location and proves paritychecker rejects it.

This closes the audit gap without widening public API, changing EMEL actor topology, or touching
`src/` machines.

## Likely Edit Surface

- `tools/paritychecker/parity_runner.cpp`
  Add canonical-path helper(s), normalize the accepted fixture path, tighten the guard, and update
  fixture-related diagnostics.
- `tools/paritychecker/parity_main.cpp`
  Replace stale "reserved contract" wording with truthful canonical-fixture wording.
- `tools/paritychecker/paritychecker_tests.cpp`
  Keep the existing success-path subprocess test, then add a negative subprocess regression for a
  same-basename file outside the canonical fixture path.

## Implementation Constraints

- Stay out of `src/` unless a hard blocker appears. This is a paritychecker-local contract fix.
- Preserve existing RTC/SML rules by avoiding state-machine topology changes; the bug is at the
  CLI/tool boundary, not inside an actor.
- Keep the test deterministic and cross-platform. A copied temporary file is safer than a symlink
  requirement.
- Prefer path normalization helpers that work on Linux, macOS, and Windows under the existing
  `PARITYCHECKER_REPO_ROOT` setup used by tests.

## Risks And Pitfalls

- `std::filesystem::canonical(...)` requires the path to exist and can throw. A safer plan is to
  compare normalized absolute paths derived from an existing canonical root plus the known relative
  fixture path, while keeping failure handling explicit.
- Test code that relies on symlinks or shell-specific temp behavior could be flaky on Windows.
- If the canonical-path helper silently falls back to basename logic, the audit gap remains open.
- Help text and runtime diagnostics must match the new contract, or the phase still leaves user-
  visible ambiguity even if the guard is stricter.

## Recommended Phase Shape

Two plans are sufficient:

1. **Canonical contract enforcement**
   Tighten the accepted generation fixture path and align CLI/runtime wording with that contract.
2. **Subprocess proof and audit hardening**
   Add regression coverage that proves canonical acceptance and basename-only rejection through the
   existing paritychecker subprocess surface.

## Validation Architecture

**Quick feedback loop**
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`

**Behavior checks**
- `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
- Negative subprocess/assertion for a same-basename non-canonical file

**Full gate**
- `scripts/paritychecker.sh`
- `scripts/quality_gates.sh`

The validation focus should stay on the external contract:
- canonical fixture path succeeds
- missing model still fails explicitly
- same-basename wrong-location fixture fails explicitly
- help text no longer describes generation as a reserved contract

## Planning Recommendation

Plan Phase 6 in **gap closure mode** conceptually, even if the command was invoked without
`--gaps`: use the audit as the hard boundary, close only `HARN-02`, and leave the broader
`llama.cpp` handoff tech debt out of scope.
