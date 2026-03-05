# Model Audit: Implicit Control Flow in Actions/Details

Date: 2026-03-03

## Scope

This audit covers `src/**/actions.hpp` and `src/**/detail.hpp`, focused on `for` loops used as
runtime control-flow substitutes rather than explicit SML guards/transitions.

Patterns audited:

- `for (size_t emel_case_* = emel_branch_*; ...)` (if/else emulation)
- `for (bool condition = ...; condition; condition = false)` (single-pass conditional blocks)

## Summary

- Total suspect loop sites: **642**
- Files affected: **19**
- In `actions.hpp`: **67** sites across **5** files
- In `detail.hpp`: **575** sites across **14** files

These loops implement runtime branching inside actions/details instead of modeling the flow
explicitly in the state machine.

## Remediation Status (This Branch)

- [x] `src/emel/gbnf/rule_parser/lexer/actions.hpp`
- [x] `src/emel/kernel/x86_64/actions.hpp`
- [x] `src/emel/kernel/aarch64/actions.hpp`
- [x] `src/emel/gbnf/rule_parser/actions.hpp`
- [x] `src/emel/token/batcher/actions.hpp`
- [x] `src/emel/docs/detail.hpp`
- [x] `src/emel/kernel/detail.hpp`
- [x] `src/emel/text/encoders/fallback/detail.hpp`
- [x] `src/emel/text/encoders/rwkv/detail.hpp`
- [x] `src/emel/text/encoders/wpm/detail.hpp`
- [x] `src/emel/text/encoders/detail.hpp`
- [x] `src/emel/text/jinja/lexer/detail.hpp`
- [x] `src/emel/text/tokenizer/preprocessor/detail.hpp`
- [x] `src/emel/batch/planner/modes/detail.hpp`
- [x] `src/emel/gbnf/rule_parser/detail.hpp`
- [x] `src/emel/text/encoders/bpe/detail.hpp`
- [x] `src/emel/text/encoders/spm/detail.hpp`
- [x] `src/emel/text/encoders/plamo2/detail.hpp`
- [x] `src/emel/text/encoders/ugm/detail.hpp`

## Reopened Findings (2026-03-04)

Status correction: the following files still contain implicit runtime control flow inside
`actions.hpp` / `detail.hpp` and must be explicitly modeled in `sm.hpp` phase/guard transitions.

### Remaining `actions.hpp` hotspots

- [x] `src/emel/token/batcher/actions.hpp` (rearchitected via explicit probe phases in `sm.hpp`)
- [x] `src/emel/gbnf/rule_parser/actions.hpp` (rearchitected with explicit
  `rule_reference_decision`, `rule_reference_shape_decision`, `quantifier_decision`, and
  `quantifier_shape_decision` phases in `sm.hpp` with explicit rule-reference
  plain/negated-shape and envelope-validity guards/actions and explicit
  `*`/`+`/`?`/braced quantifier-shape guards/actions plus explicit braced
  `exact`/`open`/`range` shape decision transitions in `sm.hpp`)
- [x] `src/emel/gbnf/rule_parser/lexer/actions.hpp` (rearchitected with explicit
  newline/rule-reference/unexpected-event branches in `sm.hpp`)
- [x] `src/emel/text/detokenizer/actions.hpp` (rearchitected with explicit
  special/byte/text decode phases and branch guards in `sm.hpp`)
- [x] `src/emel/text/encoders/rwkv/actions.hpp` (rearchitected with explicit
  `encode_validity_decision`, `encode_vocab_sync_decision`,
  `encode_capacity_decision`, `table_policy_decision`, and
  `unk_lookup_result_decision` phases in `rwkv/sm.hpp`; removed composite
  `valid_encode_and_vocab_*` and `text_non_empty_and_tables_*` guard routers and routed
  unknown-token lookup outcome and output-capacity rejection via explicit
  guards/actions instead of implicit action-finalization or loop-gated failure control)
- [x] `src/emel/text/encoders/ugm/guards.hpp` (rearchitected encode intake and
  table-policy routing into explicit `encode_validity_decision`,
  `encode_vocab_sync_decision`, and `table_policy_decision` phases in
  `text/encoders/ugm/sm.hpp`; removed composite
  `valid_encode_and_vocab_*` and `text_non_empty_and_tables_*` guard routers;
  single-use lookup/output helpers moved from `ugm/detail.hpp` into
  `ugm/actions.hpp`)
- [x] `src/emel/model/weight_loader/guards.hpp` (rearchitected apply result
  error classification into explicit `apply_request_decision`,
  `apply_error_scan_exec`, and `apply_scan_result_decision` phases in
  `model/weight_loader/sm.hpp`; removed guard-internal per-result backend-error scan loop)
- [x] `src/emel/gguf/loader/guards.hpp` (rearchitected bind/parse request
  classification into explicit `bind_request_shape_decision`,
  `bind_capacity_decision`, `parse_file_image_decision`,
  `parse_bound_storage_decision`, and `parse_capacity_decision` phases in
  `gguf/loader/sm.hpp`; removed combined validity+capacity guards as flow routers)
- [x] `src/emel/model/loader/guards.hpp` (rearchitected parse/load/validate
  policy routing into explicit `parse_phase_decision`,
  `parse_load_weights_policy_decision`, `parse_load_weights_handler_decision`,
  `load_phase_decision`, `load_map_policy_decision`, `structure_policy_decision`,
  and `architecture_policy_decision` phases in `model/loader/sm.hpp`; removed
  composite `phase_ok_and_*` guard routers)
- [x] `src/emel/text/encoders/wpm/guards.hpp` (rearchitected encode intake and
  table-policy routing into explicit `encode_validity_decision`,
  `encode_vocab_sync_decision`, and `table_policy_decision` phases in
  `text/encoders/wpm/sm.hpp`, then extended with explicit
  `encode_input_capacity_decision` routing before encode execution;
  removed composite `valid_encode_and_vocab_*` and
  `text_non_empty_and_tables_*` guard routers; removed loop-index mutation
  and loop-gated scan-stop routing from `wpm/detail.hpp`)
- [x] `src/emel/text/encoders/spm/guards.hpp` (rearchitected encode intake and
  table-policy routing into explicit `encode_validity_decision`,
  `encode_vocab_sync_decision`, and `table_policy_decision` phases in
  `text/encoders/spm/sm.hpp`, then extended with explicit
  `encode_merge_input_capacity_decision` routing before merge execution;
  removed composite `valid_encode_and_vocab_*` and
  `text_non_empty_and_tables_*` guard routers; removed loop-gated symbol-scan
  and emit error-break control from `spm/detail.hpp`; then extended with
  explicit `encode_emit_input_decision` routing in `spm/sm.hpp` for
  `symbols_present` vs `symbols_absent` before emit execution and removed
  loop-condition traversal gates (`left != -1`, `idx != -1`) from
  `spm/detail.hpp`; removed legacy `encode_spm(...)` orchestration wrapper from
  `spm/detail.hpp` so encode control flow exists only in explicit `spm/sm.hpp`
  phases/transitions)
- [x] `src/emel/text/encoders/fallback/guards.hpp` (rearchitected encode intake
  routing into explicit `encode_validity_decision` and
  `encode_vocab_sync_decision` phases in `text/encoders/fallback/sm.hpp`;
  removed composite `valid_encode_and_vocab_*` guard routers)
- [x] `src/emel/text/encoders/bpe/guards.hpp` (rearchitected encode intake,
  preprocessed-input policy, and ignore-merges/direct-word path routing into
  explicit `encode_validity_decision`, `encode_vocab_sync_decision`,
  `encode_input_policy_decision`, `encode_path_decision`, and
  `encode_direct_word_policy_decision` phases in `text/encoders/bpe/sm.hpp`,
  then extended with explicit `encode_merge_input_capacity_decision` routing
  before merge-path execution;
  removed composite `valid_encode_and_vocab_*`,
  `text_non_empty_and_{preprocessed,not_preprocessed}`, and
  `ignore_merges_fast_path`/`merge_path_required` guard routers; removed
  loop-gated symbol-scan control in `bpe/detail.hpp`)
- [x] `src/emel/text/encoders/plamo2/guards.hpp` (rearchitected encode intake
  routing into explicit `encode_validity_decision` and
  `encode_vocab_sync_decision` phases in `text/encoders/plamo2/sm.hpp`;
  removed composite `valid_encode_and_vocab_*` guard routers)
- [x] `src/emel/text/jinja/parser/lexer/actions.hpp` (rearchitected lexer token handling with
  explicit `text_boundary_candidate_decision`, `unary_candidate_decision`,
  `unary_prefix_context_decision`, `unary_prefix_allowed_decision`,
  `text_opening_block_decision` and `text_finalize_exec` with explicit opening-block trim,
  `text_trim_opening_block_result_decision` with explicit newline/zero/keep trim guards
  (no hidden `trim_text_before_opening_block` branch), leading-newline trim, and
  lstrip/rstrip branch guards,
  unary numeric-suffix decision guards, `comment_candidate_decision`,
  `comment_unterminated_exec`, `comment_finalize_exec`, `trim_prefix_candidate_decision`,
  `trim_prefix_eof_exec`, `space_eof_exec`, `mapping_close_curly_exec`,
  `mapping_candidate_decision` with explicit per-sequence mapping guards/actions (no implicit
  table lookup branch), `string_unterminated_exec`, and `string_finalize_exec` phases in
  `text/jinja/parser/lexer/sm.hpp`; removed parser-lexer-local `lookup_mapping` from
  `text/jinja/parser/lexer/detail.hpp`)

### Remaining `detail.hpp` hotspots

- [x] `src/emel/batch/planner/modes/detail.hpp` (simple + sequential + equal mode paths
  rearchitected into `modes/*/sm.hpp`; shared helpers only remain in `detail.hpp`)
- [x] `src/emel/text/encoders/ugm/detail.hpp` (rearchitected with explicit
  unknown-token, normalization, input-size, DP, and emit phases in `ugm/sm.hpp`)
- [x] `src/emel/text/encoders/rwkv/detail.hpp` (rearchitected with explicit
  unknown-token resolution phase in `rwkv/sm.hpp`; `detail::encode_rwkv` removed)
- [x] `src/emel/text/jinja/lexer/detail.hpp` (rearchitected with explicit scan phases in
  `text/jinja/parser/lexer/sm.hpp`; removed precomputed scan-plan orchestration)
- [x] `src/emel/gbnf/rule_parser/detail.hpp` (rearchitected with explicit
  `rule_reference_parse_*` and `quantifier_parse_*` phases in `rule_parser/sm.hpp`; removed
  guard-hidden parsing branches for rule references/quantifier bounds)
- [x] `src/emel/gbnf/rule_parser/lexer/detail.hpp` (rearchitected with explicit
  `next_decision` and `rule_reference_*_parse_*` phases in `rule_parser/lexer/sm.hpp`; removed
  guard-composed unknown fallback and action-only scanners from shared `detail.hpp`)

### Additional Explicit-Model Remediations (Non-loop Guard Routing)

- [x] `src/emel/memory/hybrid/sm.hpp` + `src/emel/memory/hybrid/guards.hpp`
  (rearchitected rollback-result/recurrent-error routing for `allocate_sequence`,
  `allocate_slots`, and `branch_sequence` into explicit `*_rollback_result_decision` and
  `*_recurrent_error_decision` states in `sm.hpp`; removed composite guard routers
  `rollback_accepted_and_recurrent_rejected_*` from `guards.hpp`)
- [x] `src/emel/memory/kv/sm.hpp` + `src/emel/memory/kv/guards.hpp`
  (rearchitected `allocate_slots` request classification into explicit
  `allocate_slots_request_{shape,length,block_layout,capacity}_decision` phases in `sm.hpp`;
  removed guard-internal `analyze_allocate_slots_request` routing and composite
  `allocate_slots_request_{valid,invalid,backend_error,out_of_memory}` guards)
- [x] `src/emel/memory/recurrent/sm.hpp` + `src/emel/memory/recurrent/guards.hpp`
  (rearchitected `allocate_slots` request classification into explicit
  `allocate_slots_request_{shape,length}_decision` phases and `branch_sequence`
  intake routing into explicit `branch_sequence_request_{shape,capacity}_decision`
  phases in `sm.hpp`; removed guard-internal `analyze_allocate_slots_request`
  and composite `branch_sequence_request_{valid,backend_error,invalid}` routers)

## High-Priority Findings (Action Files)

1. `src/emel/gbnf/rule_parser/lexer/actions.hpp` (24 sites, lines 40-303)
2. `src/emel/kernel/x86_64/actions.hpp` (16 sites, lines 407-692)
3. `src/emel/kernel/aarch64/actions.hpp` (16 sites, lines 326-593)
4. `src/emel/gbnf/rule_parser/actions.hpp` (10 sites, lines 242-451)
5. `src/emel/token/batcher/actions.hpp` (1 site, line 253)

Representative examples:

- `src/emel/kernel/x86_64/actions.hpp:688`
- `src/emel/gbnf/rule_parser/lexer/actions.hpp:154`
- `src/emel/gbnf/rule_parser/actions.hpp:242`
- `src/emel/token/batcher/actions.hpp:253`

## Full Inventory

| file | total | `for(bool...)` | `for(emel_case...)` |
|---|---:|---|---|
| `src/emel/text/encoders/detail.hpp` | 96 | - | 96 (110-899) |
| `src/emel/batch/planner/modes/detail.hpp` | 92 | - | 92 (29-841) |
| `src/emel/text/jinja/lexer/detail.hpp` | 70 | - | 70 (34-804) |
| `src/emel/text/tokenizer/preprocessor/detail.hpp` | 70 | - | 70 (107-641) |
| `src/emel/gbnf/rule_parser/detail.hpp` | 48 | - | 48 (27-419) |
| `src/emel/kernel/detail.hpp` | 46 | - | 46 (179-794) |
| `src/emel/text/encoders/ugm/detail.hpp` | 42 | 42 (58-559) | - |
| `src/emel/text/encoders/spm/detail.hpp` | 29 | 29 (267-566) | - |
| `src/emel/text/encoders/plamo2/detail.hpp` | 28 | 28 (69-375) | - |
| `src/emel/gbnf/rule_parser/lexer/actions.hpp` | 24 | - | 24 (40-303) |
| `src/emel/text/encoders/wpm/detail.hpp` | 19 | 19 (104-280) | - |
| `src/emel/kernel/x86_64/actions.hpp` | 16 | - | 16 (407-692) |
| `src/emel/kernel/aarch64/actions.hpp` | 16 | - | 16 (326-593) |
| `src/emel/text/encoders/bpe/detail.hpp` | 15 | 15 (266-470) | - |
| `src/emel/text/encoders/rwkv/detail.hpp` | 12 | 12 (94-193) | - |
| `src/emel/gbnf/rule_parser/actions.hpp` | 10 | - | 10 (242-451) |
| `src/emel/text/encoders/fallback/detail.hpp` | 4 | 4 (99-203) | - |
| `src/emel/docs/detail.hpp` | 4 | - | 4 (58-146) |
| `src/emel/token/batcher/actions.hpp` | 1 | - | 1 (253-253) |

## Representative Detail-Level Findings

- `src/emel/text/encoders/detail.hpp:110` and `:113` use paired `emel_case` loops for branch split.
- `src/emel/text/encoders/ugm/detail.hpp:436` uses single-pass `for (bool ...)` for early-return
  control paths.
- `src/emel/batch/planner/modes/detail.hpp:117` and `:127` use `emel_case` loops to emulate
  branch alternatives.
- `src/emel/text/tokenizer/preprocessor/detail.hpp:107` and `:110` apply the same pattern for
  vocabulary checks.
- `src/emel/gbnf/rule_parser/detail.hpp:27` and `:31` use loop-based branch emulation in builder
  operations.

## Notes

- No direct runtime `if (...)`, `switch (...)`, or `?:` were found in these files during this
  pass; branching is largely encoded through loop constructs.
- This report is intentionally focused on `for`-loop circumvention patterns per request.

## Audit Update (2026-03-04)

A comprehensive re-audit of all state machines was performed against `docs/compliance-checklist.md`, specifically looking for explicit `if`/`switch` branching and hidden error-control routing encoded in loops (`while`, `for`). 

### 1. Explicit Branching Violations
Despite earlier remediations, explicit `if` statements were found introduced/remaining in:
- `src/emel/text/tokenizer/preprocessor/detail.hpp` (18 occurrences of explicit `if` checks like `if (id >= vocab.n_tokens)`, `if (invalid_output)`, etc.)

### 2. Preprocessor Explicit-Model Progress
- [x] `src/emel/text/tokenizer/preprocessor/spm/sm.hpp` + `spm/actions.hpp` +
  `spm/guards.hpp` rearchitected with explicit request-shape phases
  (`request_buffer_decision`, `request_capacity_nonzero_decision`,
  `request_capacity_limit_decision`) and explicit parse policy phase
  (`partition_parse_special_decision`) so `parse_special` routing is modeled in
  `sm.hpp` instead of hidden inside shared partition actions/details; extended
  with explicit specials-presence and per-path input-shape decisions
  (`partition_specials_decision`, `partitioning_no_specials_input_decision`,
  `partitioning_non_bpe_parse_input_decision`,
  `partitioning_non_bpe_skip_input_decision`) and explicit
  `request_text_{empty,nonempty}` routing plus dedicated no-specials execute
  phase (`partitioning_no_specials`) so empty-input and no-specials behavior is
  modeled in `spm/sm.hpp` instead of hidden inside partition helpers.
- [x] `src/emel/text/tokenizer/preprocessor/wpm/sm.hpp` + `wpm/actions.hpp` +
  `wpm/guards.hpp` rearchitected with explicit request-shape phases
  (`request_buffer_decision`, `request_capacity_nonzero_decision`,
  `request_capacity_limit_decision`) and explicit parse policy phase
  (`partition_parse_special_decision`) so `parse_special` routing is modeled in
  `sm.hpp` instead of hidden inside shared partition actions/details; extended
  with explicit specials-presence and per-path input-shape decisions
  (`partition_specials_decision`, `partitioning_no_specials_input_decision`,
  `partitioning_non_bpe_parse_input_decision`,
  `partitioning_non_bpe_skip_input_decision`) and explicit
  `request_text_{empty,nonempty}` routing plus dedicated no-specials execute
  phase (`partitioning_no_specials`) so empty-input and no-specials behavior is
  modeled in `wpm/sm.hpp` instead of hidden inside partition helpers.
- [x] `src/emel/text/tokenizer/preprocessor/ugm/sm.hpp` + `ugm/actions.hpp` +
  `ugm/guards.hpp` rearchitected with explicit request-shape phases
  (`request_buffer_decision`, `request_capacity_nonzero_decision`,
  `request_capacity_limit_decision`) and explicit parse policy phase
  (`partition_parse_special_decision`) so `parse_special` routing is modeled in
  `sm.hpp` instead of hidden inside shared partition actions/details; extended
  with explicit specials-presence and per-path input-shape decisions
  (`partition_specials_decision`, `partitioning_no_specials_input_decision`,
  `partitioning_non_bpe_parse_input_decision`,
  `partitioning_non_bpe_skip_input_decision`) and explicit
  `request_text_{empty,nonempty}` routing plus dedicated no-specials execute
  phase (`partitioning_no_specials`) so empty-input and no-specials behavior is
  modeled in `ugm/sm.hpp` instead of hidden inside partition helpers.
- [x] `src/emel/text/tokenizer/preprocessor/rwkv/sm.hpp` + `rwkv/actions.hpp` +
  `rwkv/guards.hpp` rearchitected with explicit request-shape phases
  (`request_buffer_decision`, `request_capacity_nonzero_decision`,
  `request_capacity_limit_decision`) and explicit parse policy phase
  (`partition_parse_special_decision`) so `parse_special` routing is modeled in
  `sm.hpp` instead of hidden inside shared partition actions/details; extended
  with explicit specials-presence and per-path input-shape decisions
  (`partition_specials_decision`, `partitioning_no_specials_input_decision`,
  `partitioning_non_bpe_parse_input_decision`,
  `partitioning_non_bpe_skip_input_decision`) and explicit
  `request_text_{empty,nonempty}` routing plus dedicated no-specials execute
  phase (`partitioning_no_specials`) so empty-input and no-specials behavior is
  modeled in `rwkv/sm.hpp` instead of hidden inside partition helpers.
- [x] `src/emel/text/tokenizer/preprocessor/plamo2/sm.hpp` + `plamo2/actions.hpp` +
  `plamo2/guards.hpp` rearchitected with explicit request-shape phases
  (`request_buffer_decision`, `request_capacity_nonzero_decision`,
  `request_capacity_limit_decision`) and explicit parse policy phase
  (`partition_parse_special_decision`) so `parse_special` routing is modeled in
  `sm.hpp` instead of hidden inside shared partition actions/details.
- [x] `src/emel/text/tokenizer/preprocessor/fallback/sm.hpp` + `fallback/actions.hpp` +
  `fallback/guards.hpp` rearchitected with explicit request-shape phases
  (`request_buffer_decision`, `request_capacity_nonzero_decision`,
  `request_capacity_limit_decision`) and explicit parse policy phase
  (`partition_parse_special_decision`) so `parse_special` routing is modeled in
  `sm.hpp` instead of hidden inside shared partition actions/details; moved
  fallback partition actions out of shared `preprocessor/actions.hpp` into
  `fallback/actions.hpp` so shared files keep only shared helpers.
- [x] `src/emel/text/tokenizer/preprocessor/bpe/sm.hpp` + `bpe/actions.hpp` +
  `bpe/guards.hpp` rearchitected with explicit request-shape phases
  (`request_buffer_decision`, `request_capacity_nonzero_decision`,
  `request_capacity_limit_decision`) plus explicit specials-shape and parse policy
  phases (`partitioning_select`, `partition_parse_special_decision`) so both
  specials-presence and `parse_special` routing are modeled in `sm.hpp`; moved
  bpe-only partition helpers out of shared `preprocessor/detail.hpp` into
  `bpe/actions.hpp` per "detail only for shared helpers"; extended with explicit
  per-path empty-input decision phases
  (`partitioning_bpe_no_specials_input_decision`,
  `partitioning_bpe_with_specials_parse_input_decision`,
  `partitioning_bpe_with_specials_skip_input_decision`) and
  `request_text_{empty,nonempty}` guards in `bpe/sm.hpp` so empty-input routing is
  explicit in the model before partition execute phases; removed runtime `if`
  branching from `bpe/actions.hpp` helper flow.

### 3. Hidden Error-Control Loop Violations
The following data-plane loops terminate early on error/status checks instead of explicitly modeling the failure path in the state machine (violating the rule: "Loops in actions/detail are data-plane iteration only... not success/error/mode/retry/routing control"):
- [x] `src/emel/gbnf/rule_parser/detail.hpp` rearchitected with explicit quantifier
  braced parse-exec phases in `gbnf/rule_parser/sm.hpp`
  (`quantifier_braced_{exact,open,range}_parse_exec`), structural-only braced-shape
  guards in `guards.hpp`, and loop-control-free `parse_uint64` in `detail.hpp`
  (`std::from_chars`), removing hidden status-loop routing from parse control flow.
- [x] `src/emel/batch/planner/modes/equal/actions.hpp` rearchitected in
  `batch/planner/modes/equal/sm.hpp` with explicit
  `planning_{general,fast}_group_scan_decision` phases and
  `*_first_group_scan_{exceeds,within}_step_size` guards; removed hidden
  scan-stop loop routing (`&& !stop_group_scan`) from equal mode actions/guards.
- [x] `src/emel/text/detokenizer/actions.hpp` rearchitected in `text/detokenizer/sm.hpp` with explicit
  `decode_byte_pending_decision`, `decode_byte_pending_write`,
  `decode_text_pending_decision`, and `decode_text_pending_write` phases plus explicit
  `detokenize_pending_head_{complete,incomplete,invalid}` guards; loop-based
  `flush_pending_complete_sequences` removed.
- [x] `src/emel/text/jinja/lexer/detail.hpp` rearchitected by moving single-use
  string-escape scan helpers out of shared `jinja/lexer/detail.hpp` into
  `text/jinja/parser/lexer/actions.hpp` (per "detail only for shared helpers"),
  removing `while (ok && ...)` early-stop routing; `text/jinja/parser/lexer/sm.hpp`
  now has explicit `string_status_decision` phase for failed vs terminated vs
  unterminated string routing.
- [x] `src/emel/text/encoders/ugm/actions.hpp` + `src/emel/text/encoders/ugm/sm.hpp`
  rearchitected by splitting DP orchestration into explicit `dp_forward_exec` /
  `dp_forward_result_decision` and `dp_backtrace_exec` /
  `dp_backtrace_result_decision` phases in `ugm/sm.hpp`; removed hidden
  error-gated backtrace/emit loop routing in `ugm/actions.hpp`.
- [x] `src/emel/text/encoders/ugm/detail.hpp` normalized-input loop reworked to
  remove `while (... && state.ok)` early-stop control; loop now advances
  deterministically over input with error state carried as data and finalized
  by explicit phase status in `ugm/sm.hpp`.
- [x] `src/emel/text/encoders/plamo2/detail.hpp` rearchitected with explicit
  `table_policy_decision`, `table_sync_result_decision`, `decode_result_decision`,
  `dp_prepare_exec`, `dp_exec`, and `emit_exec` phases in
  `text/encoders/plamo2/sm.hpp`; replaced error-gated output loop condition
  (`while (pos < data_len_i32 && err == EMEL_OK)`) with deterministic
  progression and explicit phase status routing.

Remaining unchecked findings represent hidden branching control flow and must be explicitly modeled
as phase/guard transitions in their respective `sm.hpp` files.

## Additional Findings (User-Requested)

User-requested machine action/detail findings and remediation status:

- [x] `ugm/detail.hpp` (line 335) and `ugm/detail.hpp` (line 389)
  reworked to deterministic range scans (no `walking/active` loop-gated termination)
- [x] `gbnf/rule_parser/detail.hpp` (line 262)
  reworked to fixed-length deterministic hex scan (no `valid`-gated loop termination)
- [x] `gbnf/rule_parser/lexer/detail.hpp` (line 22)
  reworked to deterministic layout/reference scans (no `scan_more/skip_comment`-gated termination)
- [x] `gbnf/rule_parser/lexer/actions.hpp` (line 59)
  reworked to deterministic quoted/braced/identifier scans with explicit emit-exec phases in
  `gbnf/rule_parser/lexer/sm.hpp`
- [x] `batch/planner/modes/equal/actions.hpp` (line 33), `equal/actions.hpp` (line 91),
  `equal/actions.hpp` (line 148)
  reworked to deterministic bounded iterations (no `while`/quota-gated loop termination),
  with explicit fast/general execute-result phases in `batch/planner/modes/equal/sm.hpp`
- [x] `batch/planner/modes/sequential/actions.hpp` (line 16) and
  `sequential/actions.hpp` (line 33)
  reworked to deterministic bounded scans (no nested `while`-gated traversal), with explicit
  execute/result phases in `batch/planner/modes/sequential/sm.hpp`
- [x] `text/encoders/rwkv/actions.hpp` (`run_encode_tokens`)
  reworked to deterministic bounded traversal scans (no `while(position<...)` / `while(walk!=nullptr)`
  gating), with explicit emit-result routing phase in `text/encoders/rwkv/sm.hpp`
- [x] `text/encoders/ugm/actions.hpp` (`run_dp_forward`, `run_dp_backtrace`)
  reworked to deterministic bounded traversal scans (no `while(walking)` / `while(tracing)`
  gating), with explicit `backtrace_{ok,failed}` and `emit_{ok,failed}` decision routing in
  `text/encoders/ugm/sm.hpp`
- [x] `text/encoders/plamo2/actions.hpp` + `text/encoders/plamo2/sm.hpp`
  reworked so emit outcome is modeled explicitly in `sm.hpp` via
  `emit_result_decision` and `emit_result_{ok,failed}` guards/actions
  (`apply_emit_result_ok` / `apply_emit_result_failed`) instead of implicit
  finalization inside emit action/detail code
- [x] `text/encoders/fallback/actions.hpp` + `text/encoders/fallback/sm.hpp`
  reworked so emit outcome is modeled explicitly in `sm.hpp` via
  `emit_result_decision` and `emit_result_{ok,failed}` guards/actions
  (`apply_emit_result_ok` / `apply_emit_result_failed`) instead of implicit
  finalization inside execute/detail code
- [x] `text/renderer/actions.hpp` + `text/renderer/sm.hpp`
  reworked render post-dispatch flow into explicit `sm.hpp` phases
  (`render_commit_output_exec`, `render_strip_decision`, `render_strip_exec`,
  `render_strip_state_exec`, `render_stop_match_exec`, `render_finalize_decision`)
  and removed composite render actions that previously bundled commit/strip/stop
  behavior implicitly inside single action calls
