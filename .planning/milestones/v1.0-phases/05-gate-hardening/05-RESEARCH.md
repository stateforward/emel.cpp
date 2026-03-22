# Phase 5: Gate Hardening - Research

**Researched:** 2026-03-08
**Domain:** paritychecker generation CLI regression hardening after Phase 4
**Confidence:** HIGH

## Phase Summary

Phase 4 already pulled the generation success-path subprocess regression into
`tools/paritychecker/paritychecker_tests.cpp` and proved that the normal repo gates see it.
That closes `VER-01` early and makes the original Phase 5 wording partially stale.

The narrowest remaining Phase 5 scope is:

1. add one deterministic generation failure-path subprocess test under `paritychecker_tests`, and
2. confirm that this negative path is exercised by the existing default parity gates
   (`ctest -R paritychecker_tests`, `scripts/paritychecker.sh`, and `scripts/quality_gates.sh`).

The key recommendation is to keep Phase 5 tool-local. This should stay inside
`tools/paritychecker/` and planning docs. No `src/` SML machine changes are indicated by the
current gap, and nothing in the evidence suggests a need to reopen actor topology or generator
orchestration.

## Current State

- `.planning/REQUIREMENTS.md` marks `VER-01` complete and leaves only `VER-02` pending.
- `.planning/STATE.md` explicitly records that Phase 4 "pulled the success-path subprocess
  regression forward" so the external CLI contract is already protected.
- `.planning/phases/04-deterministic-generation-parity/04-VERIFICATION.md` states that the new
  subprocess regression in `paritychecker_tests.cpp` satisfied `VER-01` earlier than planned.
- `tools/paritychecker/paritychecker_tests.cpp` already launches
  `./paritychecker --generation --model ... --text hello --max-tokens 1`, captures stdout/stderr,
  and asserts the bounded success path through the real CLI subprocess surface.
- `tools/paritychecker/parity_runner.cpp` already publishes deterministic generation failure text
  for several negative cases, including:
  - missing model file: `generation load failed: missing model file ...`
  - wrong fixture basename: `generation requires pinned fixture ...`
  - load failure: `generation load failed (fixture=... err=...)`
  - initialize failure: `generation initialize failed (fixture=... err=...)`
  - runtime failure: `generation error (fixture=... err=...)`
  - parity drift: `generation parity mismatch (...)`
- `tools/paritychecker/parity_main.cpp` already enforces generation CLI argument validity and
  returns exit code `2` with usage text for malformed invocations.
- `tools/paritychecker/CMakeLists.txt` already registers `paritychecker_tests` as a CTest target.
- `scripts/paritychecker.sh` already builds the paritychecker tool and runs
  `ctest --test-dir "$BUILD_DIR" --output-on-failure -R paritychecker_tests`.
- `scripts/quality_gates.sh` already invokes `scripts/paritychecker.sh`, so anything added to
  `paritychecker_tests` is already part of the default repo verification path.

## Roadmap Wording Now Stale

The current Phase 5 text in `.planning/ROADMAP.md` still assumes the success-path subprocess work
is future work:

- Success criterion 1, "`paritychecker_tests` exercises the generation mode through the subprocess
  CLI path," is already true.
- Plan `05-01`, "Add subprocess generation parity coverage to `paritychecker_tests.cpp`," is also
  already true in substance.
- Success criterion 3 is partly stale. The generation slice is already visible in normal
  verification for the success path because `scripts/paritychecker.sh` and
  `scripts/quality_gates.sh` already execute `paritychecker_tests`.

The stale part is not the gate wiring anymore. The remaining gap is that the default gate only
covers the generation success path, not a generation failure path. Phase 5 should therefore be
reframed as failure-path hardening plus confirmation that the negative case rides the same default
gate.

## Likely Implementation Slices

### Slice 1: Reusable generation subprocess capture in `paritychecker_tests.cpp`

The current helper is success-path specific:

- it always includes `--generation`
- it always includes `--model`
- it always includes `--text`
- it always includes `--max-tokens 1`

That is enough for the existing positive test, but it is too rigid for narrow failure-path
coverage. The first implementation slice should generalize the helper so tests can vary generation
arguments while keeping the same cross-platform subprocess capture pattern.

Expected scope:

- keep the existing stdout/stderr file capture model
- allow a generation test to pass arbitrary extra args or omit selected args
- keep assertions based on exit code and stable stderr substrings

This is still test-only work in `tools/paritychecker/paritychecker_tests.cpp`.

### Slice 2: Add one deterministic generation failure doctest

The smallest requirement-satisfying move is one negative subprocess test that proves the generation
mode fails for the right reason.

Recommended candidate order:

1. **Wrong fixture basename using a temporary `.gguf` file**  
   Why this is strongest:
   - generation-specific, not a generic parser failure
   - deterministic and cheap
   - exercises the harness contract already enforced in `run_generation_harness_contract`
   - does not need a valid GGUF payload because basename rejection happens before file read

   Expected contract:
   - exit code `1`
   - stderr contains `generation requires pinned fixture`
   - stdout does not contain `generation parity ok`

2. **Missing model path**  
   Why this is a good fallback:
   - even narrower than the temp-file approach
   - no fixture creation needed
   - deterministic stderr already exists

   Expected contract:
   - exit code `1`
   - stderr contains `generation load failed: missing model file`
   - stdout does not contain `generation parity ok`

3. **Parser-level invalid generation invocation** such as missing `--text`, forbidden
   `--add-special`, or `--max-tokens 0`  
   Why this is weaker:
   - it only proves usage rejection
   - stderr is generic usage text rather than a generation runtime diagnostic
   - it does less to protect the actual generation harness boundary

4. **Parity mismatch path**  
   Why this should stay out of the narrow Phase 5 scope:
   - the repo rules forbid synthetic fault-injection knobs in production code
   - inducing a reliable mismatch without broadening the implementation is harder than needed for
     `VER-02`
   - invalid-input coverage already satisfies the requirement more cheaply

### Slice 3: Confirm default gate alignment, but do not widen the gate surface

No new script or CTest target is currently justified.

The existing path is already correct:

- `paritychecker_tests` is a registered CTest target
- `scripts/paritychecker.sh` runs that target
- `scripts/quality_gates.sh` runs `scripts/paritychecker.sh`

That means the alignment work is mostly verification and documentation:

- ensure the new negative doctest lives in `paritychecker_tests`
- verify it fails the target when broken
- update planning text so the phase no longer implies that success-path gate wiring is missing

## Validation Architecture

This phase needs validation across four concrete dimensions.

| Dimension | What must be proven | Preferred evidence |
|-----------|---------------------|--------------------|
| CLI subprocess contract | The negative test goes through the real `paritychecker` binary, not an in-process helper shortcut. | Doctest launches `./paritychecker --generation ...` and captures exit/stdout/stderr. |
| Failure classification | The command fails for a specific, stable reason rather than any non-zero exit. | Assert exit code and a stable stderr substring tied to the expected failure. |
| Success-path non-regression | Phase 4's bounded success path still passes unchanged. | Keep the existing success doctest and run it beside the new negative doctest. |
| Default gate visibility | The new negative slice runs under the standard repo verification entrypoints. | `ctest -R paritychecker_tests`, `scripts/paritychecker.sh`, and `scripts/quality_gates.sh`. |

Recommended validation commands for the implementation phase:

- Quick loop: `scripts/paritychecker.sh`
- Focused target: `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
- Full repo gate: `scripts/quality_gates.sh`

Concrete assertions to require:

- Positive generation subprocess test:
  - exit code `0`
  - stdout contains `generation parity ok`
  - stderr is empty
- Negative generation subprocess test:
  - exit code `1` for harness/runtime failures, or `2` only if the phase deliberately chooses a
    parser-usage case
  - stderr contains the expected stable reason substring
  - stdout does not contain the success marker

## Risks

- **Testing the wrong seam:** a parser usage error is easy to add, but it under-covers the actual
  generation harness boundary.
- **Brittle stderr assertions:** full-path comparisons or full usage-text snapshots will be noisy
  across environments. Match stable substrings instead.
- **Scope creep into mismatch injection:** adding a parity-mismatch test could pull in artificial
  knobs or deeper runtime changes that the repo rules explicitly discourage.
- **Accidentally reopening `src/` topology:** nothing in the current evidence suggests Phase 5
  needs generator or loader machine edits. Keep the work in `tools/paritychecker/` unless a new
  failing test proves otherwise.
- **Cross-platform temp-file behavior:** if the wrong-fixture test uses a temp file, cleanup and
  quoting must follow the same Windows/POSIX subprocess conventions already present in
  `paritychecker_tests.cpp`.

## Recommended Plan Split

### Recommended 05-01

**Name:** Add deterministic generation failure subprocess coverage

Scope:

- generalize the generation subprocess capture helper in `paritychecker_tests.cpp`
- add one negative doctest
- prefer wrong-fixture temp-file coverage; use missing-model coverage as fallback if that proves
  materially simpler

Success condition:

- `VER-02` is satisfied by automated subprocess coverage with reason-specific assertions

### Recommended 05-02

**Name:** Align phase wording and verify the default parity gate

Scope:

- update stale planning language so Phase 5 no longer claims success-path subprocess coverage is
  still missing
- rerun the standard paritychecker and repo gates
- verify the new negative test is visible through the existing gate chain without adding a new one

Success condition:

- the planning docs describe the real remaining gap accurately
- the default gate path proves both generation success and one expected failure

## Recommended Scope Boundary

Phase 5 should be treated as a narrow hardening pass, not a new runtime feature phase.

In scope:

- `tools/paritychecker/paritychecker_tests.cpp`
- possibly tiny supporting adjustments in `tools/paritychecker/parity_main.cpp` or
  `tools/paritychecker/parity_runner.cpp` only if the new failing test exposes an unstable
  diagnostic contract
- planning text updates that remove stale Phase 5 wording

Out of scope for the narrow completion of `VER-02`:

- new `src/` machine behavior
- new public API surface
- synthetic parity-mismatch injection mechanisms
- broad generation matrices or multiple additional failure classes

## Bottom Line

Phase 4 already completed the success-path half of Gate Hardening. The smallest remaining Phase 5
is a one-negative-test hardening pass plus roadmap/gate alignment cleanup. If that negative test
lives in `paritychecker_tests.cpp`, the default parity verification path already does the rest.
