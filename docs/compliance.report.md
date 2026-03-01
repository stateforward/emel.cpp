# Compliance Report

- Generated: 2026-03-01
- Checklist reference: `docs/compliance-checklist.md`
- Scope: `src/emel/text/encoders` (7 encoder state machines + shared encoder orchestration files)
- Result: **FAIL** (1 merge-blocking checklist item remains in 1/7 encoders)

## Audit Summary

- Machines audited: `7`
  - `bpe`, `fallback`, `plamo2`, `rwkv`, `spm`, `ugm`, `wpm`
- Structural SML checks (destination-first rows, wrappers, explicit unexpected-event handling): **PASS**
- Event/context/error orchestration checks: **PASS** after this refactor
- Action/guard architecture checks: **FAIL** on strict checklist item `2.5` for `fallback`; **PASS** for `bpe`, `plamo2`, `rwkv`, `spm`, `ugm`, `wpm`

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
4. BPE encoder call-path no longer depends on shared branching detail helpers.
   - BPE table-prep action now calls local `ensure_bpe_tables`: `src/emel/text/encoders/bpe/actions.hpp:31-36`
   - BPE path guard now calls local `bpe_lookup_token`: `src/emel/text/encoders/bpe/guards.hpp:75-79`
   - BPE local kernels now provide table build/lookups/symbol/push operations in component detail:
     `src/emel/text/encoders/bpe/detail.hpp:19-339`
   - No `if`/`else`/`switch` tokens remain in BPE `actions.hpp`, `guards.hpp`, `detail.hpp`, or `sm.hpp` (regex validation pass).
5. PLaMo2 encoder call-path now uses local branchless kernels in component detail.
   - Runtime encode action still dispatches via `plamo2::detail::encode_plamo2`:
     `src/emel/text/encoders/plamo2/actions.hpp:33-38`
   - PLaMo2 local detail now owns token text lookup, byte token parsing, push path, table build, and encode kernels:
     `src/emel/text/encoders/plamo2/detail.hpp:20-388`
   - No `if`/`else`/`switch`/`?` tokens remain in PLaMo2 `actions.hpp`, `guards.hpp`, `detail.hpp`, or `sm.hpp` (regex validation pass).
6. RWKV encoder now models table-sync behavior as explicit machine phases and keeps encode-kernel execution phase-local.
   - Added explicit table-sync states and transitions:
     `src/emel/text/encoders/rwkv/sm.hpp` (`table_sync_exec`, `table_sync_result_decision`)
   - Added table-sync action and table-ready/table-missing guards:
     `src/emel/text/encoders/rwkv/actions.hpp`, `src/emel/text/encoders/rwkv/guards.hpp`
   - `encode_rwkv` now requires prepared tables and no longer performs hidden table construction:
     `src/emel/text/encoders/rwkv/detail.hpp:199-237`
   - Added regression test for this contract:
     `tests/text/encoders/rwkv_tests.cpp` (`encoder_rwkv_encode_requires_prepared_tables`)
7. SPM encoder now models table-sync + explicit prepare/merge/emit phases and keeps each phase action-local.
   - Added explicit prepare and merge phase states/transitions:
     `src/emel/text/encoders/spm/sm.hpp` (`encode_prepare_exec`, `encode_prepare_result_decision`,
     `encode_merge_exec`, `encode_merge_result_decision`)
   - Added dedicated phase actions:
     `src/emel/text/encoders/spm/actions.hpp` (`run_prepare`, `run_merge`, `run_encode`)
   - Split monolithic detail encode path into explicit phase kernels:
     `src/emel/text/encoders/spm/detail.hpp` (`prepare_spm`, `merge_spm`, `emit_spm`)
8. UGM encoder now models table-sync as explicit machine phases and uses branchless local detail kernels.
   - Added explicit table-sync states and transitions:
     `src/emel/text/encoders/ugm/sm.hpp` (`table_sync_exec`, `table_sync_result_decision`)
   - Added UGM table-sync action and table-ready/table-missing guards:
     `src/emel/text/encoders/ugm/actions.hpp`, `src/emel/text/encoders/ugm/guards.hpp`
   - `encode_ugm` now requires prepared tables for non-empty text and no longer performs hidden table construction:
     `src/emel/text/encoders/ugm/detail.hpp`
   - UGM detail call-path no longer depends on shared branching helpers (`token_text`, `push_token`, `utf8_len`) and no longer
     performs dispatch-time `std::string` copy during normalization:
     `src/emel/text/encoders/ugm/detail.hpp`
   - Added regression test for this contract:
     `tests/text/encoders/ugm_tests.cpp` (`encoder_ugm_encode_requires_prepared_tables`)
9. WPM encoder now models table-sync as explicit machine phases and uses branchless local detail kernels.
   - Added explicit table-sync states and transitions:
     `src/emel/text/encoders/wpm/sm.hpp` (`table_sync_exec`, `table_sync_result_decision`)
   - Added WPM table-sync action and table-ready/table-missing guards:
     `src/emel/text/encoders/wpm/actions.hpp`, `src/emel/text/encoders/wpm/guards.hpp`
   - `encode_wpm` now requires prepared tables for non-empty text and no longer performs hidden table construction:
     `src/emel/text/encoders/wpm/detail.hpp`
   - WPM detail call-path no longer depends on shared branching helpers (`ensure_tables`, `lookup_token`, `push_token`):
     `src/emel/text/encoders/wpm/detail.hpp`
   - Added regression test for this contract:
     `tests/text/encoders/wpm_tests.cpp` (`encoder_wpm_encode_requires_prepared_tables`)

## Remaining Non-Compliance (Merge-Blocking)

### 1) Runtime conditionals in functions called from actions (Checklist Section 2, item 2.5)

Checklist mapping:
- `2) Action and Guard Architecture`
- `Runtime control-flow conditionals are not implemented in functions called from actions`

Evidence:
- Actions call variant detail kernels from `run_encode`:
  - `src/emel/text/encoders/fallback/actions.hpp:41`
- These detail kernels contain runtime branching (`if (...)`) for algorithmic encoding behavior:
  - `src/emel/text/encoders/fallback/detail.hpp`

## Per-Machine Compliance Status

- `src/emel/text/encoders/bpe/sm.hpp`: **PASS**
- `src/emel/text/encoders/plamo2/sm.hpp`: **PASS** (checked off)
- `src/emel/text/encoders/fallback/sm.hpp`: **FAIL** (item 2.5 only)
- `src/emel/text/encoders/rwkv/sm.hpp`: **PASS** (checked off)
- `src/emel/text/encoders/spm/sm.hpp`: **PASS**
- `src/emel/text/encoders/ugm/sm.hpp`: **PASS**
- `src/emel/text/encoders/wpm/sm.hpp`: **PASS**

## Validation Commands Used

- `scripts/quality_gates.sh` (passes in current workspace; benchmark snapshot warnings are pre-existing baseline issues)
- `rg -n "\\bif\\b|\\belse\\b|\\bswitch\\b|\\?" src/emel/text/encoders/bpe/actions.hpp src/emel/text/encoders/bpe/guards.hpp src/emel/text/encoders/bpe/detail.hpp src/emel/text/encoders/bpe/sm.hpp` (no matches)
- `rg -n "\\bif\\b|\\belse\\b|\\bswitch\\b|\\?" src/emel/text/encoders/plamo2/actions.hpp src/emel/text/encoders/plamo2/guards.hpp src/emel/text/encoders/plamo2/detail.hpp src/emel/text/encoders/plamo2/sm.hpp` (no matches)
- `rg -n "\\bif\\b|\\belse\\b|\\bswitch\\b|\\?" src/emel/text/encoders/rwkv/actions.hpp src/emel/text/encoders/rwkv/guards.hpp src/emel/text/encoders/rwkv/detail.hpp src/emel/text/encoders/rwkv/sm.hpp` (no matches)
- `rg -n "\\bif\\b|\\belse\\b|\\bswitch\\b|\\?" src/emel/text/encoders/ugm/actions.hpp src/emel/text/encoders/ugm/guards.hpp src/emel/text/encoders/ugm/detail.hpp src/emel/text/encoders/ugm/sm.hpp` (no matches)
- `rg -n "encoders::detail::token_text|encoders::detail::push_token|encoders::detail::utf8_len|\\.insert\\(|\\.traverse\\(" src/emel/text/encoders/ugm/detail.hpp` (no matches)
- `rg -n "std::string input\\(" src/emel/text/encoders/ugm/detail.hpp` (no matches)
- `rg -n "\\bif\\b|\\belse\\b|\\bswitch\\b|\\?" src/emel/text/encoders/wpm/actions.hpp src/emel/text/encoders/wpm/guards.hpp src/emel/text/encoders/wpm/detail.hpp src/emel/text/encoders/wpm/sm.hpp` (no matches)
- `rg -n "encoders::detail::ensure_tables|encoders::detail::lookup_token|encoders::detail::push_token" src/emel/text/encoders/wpm/detail.hpp` (no matches)
- `zig c++ ... /tmp/rwkv_compile_check.cpp` (RWKV `sm.hpp` compile smoke passes)
- `zig c++ ... /tmp/rwkv_behavior_check.cpp && /tmp/rwkv_behavior_check` (unprepared-table failure and prepared-table success contract passes)
- `rg -n "\\bif \\(" src/emel/text/encoders/fallback/detail.hpp` (matches remain)
- `rg -n "\\bif\\b|\\belse\\b|\\bswitch\\b|\\?" src/emel/text/encoders/spm/actions.hpp src/emel/text/encoders/spm/guards.hpp src/emel/text/encoders/spm/detail.hpp src/emel/text/encoders/spm/sm.hpp` (no matches)
