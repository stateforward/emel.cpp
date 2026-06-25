---
phase: 233-public-loader-and-maintained-entrypoints
verified: 2026-05-08T00:10:00.000Z
status: verified
requirements_touched:
  - PUB-01
  - PUB-02
  - PUB-03
  - PUB-04
  - PUB-05
authority:
  - manager_validation
  - phase233_navigator_review
---

# Phase 233: Public loader and maintained entrypoints — Verification

**Status: verified** — **`PUB-01`–`PUB-05`** are **Satisfied** for Phase 233 per **manager validation** and
**phase233-navigator final review PASS** (peer review of source/test evidence; no external message ID required).
Source is **PASS**; this ledger summarizes observable proof without restating implementation patches.

## Observable truths

| # | Evidence (source-backed) |
|---|--------------------------|
| 1 | **Public `staged_read` contract + dispatch in `io::loader`:** single-tensor **`staged_window`** and batch **`staged_window_batch`** dispatch via injected **`io_staged_read`** (`src/emel/io/loader/sm.hpp`, `actions.hpp`, `context.hpp`). |
| 2 | **Maintained benchmarks / fixtures wire public `io_staged_read`:** **`generation_bench`**, **`sortformer_fixture`**, **`parity_engines`**, **`embedded_size`** probe paths updated so **`io_staged_read` actors** are supplied where staged strategy applies (closeout review; not detail reach-through). |
| 3 | **`model/loader` storage-backed `staged_read` route:** doctest coverage in **`tests/model/loader/lifecycle_tests.cpp`** exercises **`strategy_kind::staged_read`** with real **`emel::io::staged_read::sm`** in **`io::loader` context** (`io_staged_read`), plus static source scans rejecting **`emel/io/staged_read/{actions,detail,guards}.hpp`** includes where forbidden. |
| 4 | **Tool / private-include hygiene:** tests and source scans guard against forbidden **`staged_read` detail/actions/guards** includes in maintained surfaces; the **no tool-local staged-buffer duplication** finding is from navigator + source-scan review over maintained tool entrypoints. |
| 5 | **Timing snapshot:** **`snapshots/quality_gates/timing.txt`** matches **`HEAD`** in a clean working-tree check for timing (no accidental gate churn carried in closeout). |

## Automated regression (post `staged_read` sm/actions clarity cleanup)

**Milestone worktree, 2026-05-08** — manager rerun after the cleanup (not stale pre-change validation):

| Step | Command | Outcome |
|------|---------|---------|
| Build | `ninja -C build emel_tests_bin` | Exit **0** (**`no work to do`**) |
| Tests | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` | **Passed**, **2/2** (total **~2.79s**) |

## Quality gates (Phase 233)

- **`scripts/quality_gates.sh`** for a Phase 233 changed-file corpus: **not run** in this driver slice →
  **no scoped quality gate pass claimed** for Phase 233. (Phase **232** residual **exit 2** story remains
  in **`232-VERIFICATION.md`** — unrelated ledger.)

## Residual / explicitly not claimed

- Full-repo **`scripts/quality_gates.sh` PASS** for Phase 233 (pending / not executed here).
- Any waiver of Phase **232** bench/parity **exit 2** regressions documented there.

## Result

**`PUB-01`–`PUB-05`** are **Satisfied** at the Phase **233** milestone layer with public-contract-only
access to **`staged_read`** through **`io::loader`** and maintained entrypoints as cited above; **`TST-01` /
`TST-02`** remain Phase **234** work.
