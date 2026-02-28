# Compliance Report

- Generated: 2026-02-28
- Checklist reference: `docs/compliance-checklist.md`
- Scope: `src/emel/text/encoders` (7 encoder state machines + shared encoder orchestration files)
- Result: **FAIL** (1 merge-blocking checklist item remains)

## Audit Summary

- Machines audited: `7`
  - `bpe`, `fallback`, `plamo2`, `rwkv`, `spm`, `ugm`, `wpm`
- Structural SML checks (destination-first rows, wrappers, explicit unexpected-event handling): **PASS**
- Event/context/error orchestration checks: **PASS** after this refactor
- Action/guard architecture checks: **FAIL** on strict checklist item `2.5`

## Fixed In This Pass

1. Removed shared enum error typing from encoder orchestration runtime state.
   - `event::encode_ctx.err` is now `int32_t` boundary error code: `src/emel/text/encoders/events.hpp:36-39`
   - `events::encoding_error.err` is now `int32_t`: `src/emel/text/encoders/events.hpp:55-58`
2. Shared action/guard/detail orchestration now uses boundary error codes (`EMEL_*`) instead of shared enum control state.
   - `src/emel/text/encoders/actions.hpp:23-61`
   - `src/emel/text/encoders/guards.hpp:27-36`
   - `src/emel/text/encoders/detail.hpp:45-83`
3. All 7 machine wrappers now derive success/failure from runtime error code and store `last_error_` as `int32_t`.
   - Example: `src/emel/text/encoders/bpe/sm.hpp:159-180`
   - Same wrapper pattern confirmed in all seven encoder `sm.hpp` files.

## Remaining Non-Compliance (Merge-Blocking)

### 1) Runtime conditionals in functions called from actions (Checklist Section 2, item 2.5)

Checklist mapping:
- `2) Action and Guard Architecture`
- `Runtime control-flow conditionals are not implemented in functions called from actions`

Evidence:
- Actions call variant detail kernels from `run_encode`:
  - `src/emel/text/encoders/bpe/actions.hpp:31`
  - `src/emel/text/encoders/fallback/actions.hpp:31`
  - `src/emel/text/encoders/plamo2/actions.hpp:35`
  - `src/emel/text/encoders/rwkv/actions.hpp:33`
  - `src/emel/text/encoders/spm/actions.hpp:31`
  - `src/emel/text/encoders/ugm/actions.hpp:34`
  - `src/emel/text/encoders/wpm/actions.hpp:31`
- These detail kernels contain runtime branching (`if (...)`) for algorithmic encoding behavior:
  - `src/emel/text/encoders/bpe/detail.hpp`
  - `src/emel/text/encoders/fallback/detail.hpp`
  - `src/emel/text/encoders/plamo2/detail.hpp`
  - `src/emel/text/encoders/rwkv/detail.hpp`
  - `src/emel/text/encoders/spm/detail.hpp`
  - `src/emel/text/encoders/ugm/detail.hpp`
  - `src/emel/text/encoders/wpm/detail.hpp`

## Per-Machine Compliance Status

- `src/emel/text/encoders/bpe/sm.hpp`: **FAIL** (item 2.5 only)
- `src/emel/text/encoders/fallback/sm.hpp`: **FAIL** (item 2.5 only)
- `src/emel/text/encoders/plamo2/sm.hpp`: **FAIL** (item 2.5 only)
- `src/emel/text/encoders/rwkv/sm.hpp`: **FAIL** (item 2.5 only)
- `src/emel/text/encoders/spm/sm.hpp`: **FAIL** (item 2.5 only)
- `src/emel/text/encoders/ugm/sm.hpp`: **FAIL** (item 2.5 only)
- `src/emel/text/encoders/wpm/sm.hpp`: **FAIL** (item 2.5 only)

## Validation Commands Used

- `scripts/quality_gates.sh` (passes in current workspace; benchmark snapshot warnings are pre-existing baseline issues)
- `rg` checks over `src/emel/text/encoders/**` for transition syntax, unexpected-event handling, and runtime branching patterns
