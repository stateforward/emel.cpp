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
- [x] `src/emel/model/weight_loader/sm.hpp` + `src/emel/model/weight_loader/guards.hpp`
  (rearchitected callback-phase routing to explicit bind/plan/apply error-class guards
  in `model/weight_loader/sm.hpp`; removed generic `*_phase_ok` / `*_phase_failed`
  guard routing and replaced it with explicit `error::none`, typed runtime error classes,
  untracked, and unknown branches)
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
- [x] `src/emel/text/tokenizer/sm.hpp` + `src/emel/text/tokenizer/guards.hpp`
  (rearchitected bind-phase callback routing into explicit
  `bind_preprocessor_error_*` and `bind_encoder_error_*` decision guards in
  `sm.hpp`; removed generic `phase_ok` / `phase_failed` routing from tokenizer
  bind flow and made runtime error classes explicit, including unknown error
  fallback)
- [x] `src/emel/text/detokenizer/sm.hpp` + `src/emel/text/detokenizer/guards.hpp`
  (rearchitected bind and detokenize decision routing to explicit typed
  `bind_error_*` / `detokenize_error_*` guard branches in `sm.hpp`; removed
  generic phase routing guards and modeled `none`, typed error classes,
  untracked, and unknown paths explicitly)
- [x] `src/emel/gguf/loader/sm.hpp` + `src/emel/gguf/loader/guards.hpp`
  (rearchitected probe/bind/parse outcome dispatch to explicit typed
  `probe_error_*`, `bind_error_*`, and `parse_error_*` guard branches in
  `sm.hpp`; removed generic `*_phase_ok` / `*_phase_failed` routing and modeled
  `none`, typed error classes, untracked, and unknown paths explicitly)
- [x] `src/emel/graph/allocator/sm.hpp` + `src/emel/graph/allocator/guards.hpp`
  (rearchitected allocation finalization routing to explicit typed
  `allocation_error_*` guard branches in `sm.hpp`; removed generic
  `phase_ok` / `phase_failed` routing and modeled `none`, typed error classes,
  untracked, and unknown paths explicitly)
- [x] `src/emel/graph/processor/sm.hpp` + `src/emel/graph/processor/guards.hpp`
  (rearchitected execution finalization routing to explicit typed
  `execution_error_*` guard branches in `sm.hpp`; removed generic
  `phase_ok` / `phase_failed` routing and modeled `none`, typed error classes,
  untracked, and unknown paths explicitly)
- [x] `src/emel/graph/sm.hpp` + `src/emel/graph/guards.hpp`
  (rearchitected compute finalization routing to explicit typed
  `compute_error_*` guard branches in `sm.hpp`; removed generic
  `compute_phase_ok` / `compute_phase_failed` routing and modeled `none`,
  typed error classes, untracked, and unknown paths explicitly)
- [x] `src/emel/graph/assembler/sm.hpp` + `src/emel/graph/assembler/guards.hpp`
  (rearchitected reserve and assemble finalization routing to explicit typed
  `reserve_error_*` and `assemble_error_*` guard branches in `sm.hpp`; removed
  generic `reserve_phase_*` / `assemble_phase_*` routing and modeled `none`,
  typed error classes, untracked, and unknown paths explicitly)
- [x] `src/emel/text/jinja/parser/program_parser/sm.hpp` +
  `src/emel/text/jinja/parser/program_parser/guards.hpp` (rearchitected statement/expression
  submachine result routing to explicit typed `parse_error_*` guard branches in `sm.hpp`;
  removed generic `phase_ok` / `phase_failed` routing and modeled `none`, typed error
  classes, untracked, and unknown paths explicitly)
- [x] `src/emel/text/jinja/parser/sm.hpp` +
  `src/emel/text/jinja/parser/guards.hpp` (rearchitected tokenize/classify/final parse routing
  to explicit typed `parse_error_*` guard branches in `sm.hpp`; removed generic `phase_ok` /
  `phase_failed` routing and modeled `none`, typed error classes, untracked, and unknown
  paths explicitly)
- [x] `src/emel/text/jinja/parser/lexer/sm.hpp` +
  `src/emel/text/jinja/parser/lexer/guards.hpp` (rearchitected scan-result routing to explicit
  typed `parse_error_*` guard branches in `sm.hpp`; removed generic
  `phase_failed`/`phase_has_token`/`phase_at_eof`/`phase_unhandled` routing and modeled explicit
  error-class, token-available, no-token-eof, and unhandled paths)
- [x] `src/emel/gbnf/rule_parser/sm.hpp` + `src/emel/gbnf/rule_parser/guards.hpp`
  (rearchitected literal/character-class consume result routing and final parse dispatch routing
  to explicit typed `parse_error_*` guard branches in `sm.hpp`; removed generic `phase_ok` /
  `phase_failed` routing and modeled `none`, typed error classes, untracked, and unknown paths
  explicitly)

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
  `sm.hpp` instead of hidden inside shared partition actions/details; extended
  with explicit specials-presence and per-path input-shape decisions
  (`partition_specials_decision`, `partitioning_no_specials_input_decision`,
  `partitioning_non_bpe_parse_input_decision`,
  `partitioning_non_bpe_skip_input_decision`) and explicit
  `request_text_{empty,nonempty}` routing plus dedicated no-specials execute
  phase (`partitioning_no_specials`) so empty-input and no-specials behavior is
  modeled in `plamo2/sm.hpp` instead of hidden inside partition helpers.
- [x] `src/emel/text/tokenizer/preprocessor/fallback/sm.hpp` + `fallback/actions.hpp` +
  `fallback/guards.hpp` rearchitected with explicit request-shape phases
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
  modeled in `fallback/sm.hpp` instead of hidden inside partition helpers; moved
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
- [x] `token/batcher/actions.hpp` + `token/batcher/sm.hpp` + `token/batcher/guards.hpp`
  reworked request validation routing into explicit `sm.hpp` phases
  (`request_outputs_decision`, `request_token_counts_decision`,
  `request_capacities_decision`, `request_token_ids_decision`,
  `request_seq_payload_decision`) with explicit invalid-request guards/actions;
  removed composite `probe_request_validity` request classifier from
  `token/batcher/actions.hpp` so request validation control flow is modeled by
  explicit guarded transitions.
- [x] `token/batcher/sm.hpp` + `token/batcher/guards.hpp`
  reworked probe outcome routing into explicit `sm.hpp` branches for
  `positions_seeded_probe`, `positions_unseeded_probe`, `single_output_probe`,
  and `continuity_probe` using dedicated outcome guards
  (`*_probe_ok`, `*_probe_invalid_request`, `positions_seeded_probe_backend_error`)
  plus explicit invalid/backend/internal error transitions instead of generic
  `phase_ok` / `phase_failed` routing.
- [x] `text/encoders/bpe/sm.hpp` + `text/encoders/bpe/guards.hpp`
  reworked table-prepare and encode-result routing into explicit error-class
  branches (`*_ok`, `*_invalid_argument_error`, `*_backend_error`,
  `*_model_invalid_error`, `*_unknown_error`) so `bpe/sm.hpp` does not rely on
  generic `phase_ok` / `phase_failed` guards for action/detail-driven outcomes.
- [x] `text/encoders/wpm/sm.hpp` + `text/encoders/wpm/guards.hpp`
  reworked table-sync and encode-result routing into explicit error-class
  branches (`*_ok`, `*_invalid_argument_error`, `*_backend_error`,
  `*_model_invalid_error`, `*_unknown_error`) so `wpm/sm.hpp` does not rely on
  generic `phase_ok` / `phase_failed` guards for action/detail-driven outcomes.

## Reopened Findings (2026-03-05)

- [x] `src/emel/text/tokenizer/preprocessor/detail.hpp`
  refactored shared preprocessor helper paths to explicit step-dispatch helpers
  (no runtime `if`/`for`/`while`/`?:` branch statements in the shared
  build/partition logic); machine-level branch routing remains explicit in
  preprocessor `sm.hpp` guards/states (`no_specials`/`has_specials`,
  `parse_special_enabled`/`parse_special_disabled`, and explicit partition
  result error-class guards).
- [x] `src/emel/generator/sm.hpp` + `src/emel/generator/guards.hpp`
  reworked conditioning/planning/prefill/decode phase routing away from generic
  `phase_ok` / `phase_failed` and into explicit error-class branches
  (`phase_none`, `phase_invalid_request_error`, `phase_backend_error`,
  `phase_unknown_error`) plus explicit final error-channel routing via
  `generate_error_channel_decision`.
- [x] `src/emel/model/loader/sm.hpp` + `src/emel/model/loader/guards.hpp`
  replaced generic phase routing (`phase_ok` / `phase_failed`) with explicit
  error-class transitions (`error_none`, `error_invalid_request`,
  `error_parse_failed`, `error_backend_error`, `error_model_invalid`,
  `error_internal_error`, `error_untracked`, `error_unknown`) across parse/load/
  map/validation decision phases.
- [x] `src/emel/text/jinja/parser/lexer/actions.hpp` +
  `src/emel/text/jinja/parser/lexer/sm.hpp`
  reworked scanner flow by removing loop-gated scan/trim actions and adding
  explicit string-scan phases (`string_scan_exec`,
  `string_content_scan_exec`, `string_materialize_exec`) in `sm.hpp`; replaced
  loop-based text/comment/space/word boundary traversals with explicit
  boundary-probe actions.
- [x] `src/emel/text/renderer/actions.hpp` + `src/emel/text/renderer/sm.hpp` +
  `src/emel/text/renderer/guards.hpp`
  reworked render-strip flow into explicit `render_strip_prefix_scan_exec`,
  `render_strip_prefix_decision`, and `render_strip_apply_exec` phases with
  explicit `strip_prefix_nonzero` / `strip_prefix_zero` guards; removed
  loop-gated strip traversal from renderer action code.
- [x] `src/emel/gbnf/rule_parser/actions.hpp` +
  `src/emel/gbnf/rule_parser/sm.hpp`
  reworked literal/character-class consume routing into explicit
  `literal_consume_exec`/`literal_consume_result_decision` and
  `character_class_consume_exec`/`character_class_consume_result_decision`
  phases in `sm.hpp`; removed loop-driven literal/char-class traversal from
  action code.
- [x] `src/emel/text/tokenizer/preprocessor/bpe/sm.hpp` +
  `src/emel/text/tokenizer/preprocessor/bpe/guards.hpp`
  reworked build-specials and partition result routing into explicit
  error-class branches (`*_ok`, `*_invalid_request_error`,
  `*_backend_error`, `*_unknown_error`) so `bpe/sm.hpp` no longer relies on
  generic `phase_ok` / `phase_failed` guards.
- [x] `src/emel/text/tokenizer/preprocessor/spm/sm.hpp` +
  `src/emel/text/tokenizer/preprocessor/spm/guards.hpp`
  reworked build-specials and partition result routing into explicit
  error-class branches (`*_ok`, `*_invalid_request_error`,
  `*_backend_error`, `*_unknown_error`) so `spm/sm.hpp` no longer relies on
  generic `phase_ok` / `phase_failed` guards.
- [x] `src/emel/text/tokenizer/preprocessor/rwkv/sm.hpp` +
  `src/emel/text/tokenizer/preprocessor/rwkv/guards.hpp`
  reworked build-specials and partition result routing into explicit
  error-class branches (`*_ok`, `*_invalid_request_error`,
  `*_backend_error`, `*_unknown_error`) so `rwkv/sm.hpp` no longer relies on
  generic `phase_ok` / `phase_failed` guards.
- [x] `src/emel/text/tokenizer/preprocessor/ugm/sm.hpp` +
  `src/emel/text/tokenizer/preprocessor/ugm/guards.hpp`
  reworked build-specials and partition result routing into explicit
  error-class branches (`*_ok`, `*_invalid_request_error`,
  `*_backend_error`, `*_unknown_error`) so `ugm/sm.hpp` no longer relies on
  generic `phase_ok` / `phase_failed` guards.
- [x] `src/emel/text/tokenizer/preprocessor/wpm/sm.hpp` +
  `src/emel/text/tokenizer/preprocessor/wpm/guards.hpp`
  reworked build-specials and partition result routing into explicit
  error-class branches (`*_ok`, `*_invalid_request_error`,
  `*_backend_error`, `*_unknown_error`) so `wpm/sm.hpp` no longer relies on
  generic `phase_ok` / `phase_failed` guards.
- [x] `src/emel/text/tokenizer/preprocessor/fallback/sm.hpp` +
  `src/emel/text/tokenizer/preprocessor/fallback/guards.hpp`
  reworked build-specials and partition result routing into explicit
  error-class branches (`*_ok`, `*_invalid_request_error`,
  `*_backend_error`, `*_unknown_error`) so `fallback/sm.hpp` no longer relies
  on generic `phase_ok` / `phase_failed` guards.
- [x] `src/emel/text/tokenizer/preprocessor/plamo2/sm.hpp` +
  `src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp`
  reworked build-specials and partition result routing into explicit
  error-class branches (`*_ok`, `*_invalid_request_error`,
  `*_backend_error`, `*_unknown_error`) so `plamo2/sm.hpp` no longer relies on
  generic `phase_ok` / `phase_failed` guards.

- [x] `text/encoders/spm/sm.hpp` + `text/encoders/spm/guards.hpp`
  reworked table-sync, prepare, merge, and final encode-result routing into
  explicit error-class branches (`*_ok`, `*_invalid_argument_error`,
  `*_backend_error`, `*_model_invalid_error`, `*_unknown_error`) so `spm/sm.hpp`
  no longer relies on generic `phase_ok` / `phase_failed` guards for
  action/detail-driven outcomes.
- [x] `text/encoders/rwkv/sm.hpp` + `text/encoders/rwkv/guards.hpp`
  reworked table-sync and final encode-result routing into explicit error-class
  branches (`*_ok`, `*_invalid_argument_error`, `*_backend_error`,
  `*_model_invalid_error`, `*_unknown_error`) so `rwkv/sm.hpp` no longer relies
  on generic `phase_ok` / `phase_failed` guards for action/detail-driven
  outcomes.
- [x] `text/encoders/fallback/sm.hpp` + `text/encoders/fallback/guards.hpp`
  reworked table-prepare and final encode-result routing into explicit
  error-class branches (`*_ok`, `*_invalid_argument_error`, `*_backend_error`,
  `*_model_invalid_error`, `*_unknown_error`) so `fallback/sm.hpp` no longer
  relies on generic `phase_ok` / `phase_failed` guards for action/detail-driven
  outcomes.
- [x] `text/encoders/ugm/sm.hpp` + `text/encoders/ugm/guards.hpp`
  reworked table-sync, normalization, input-prepare, and DP-forward result
  routing into explicit error-class branches (`*_ok`,
  `*_invalid_argument_error`, `*_backend_error`, `*_model_invalid_error`,
  `*_unknown_error`) so `ugm/sm.hpp` no longer relies on generic
  `phase_ok` / `phase_failed` guards for action/detail-driven outcomes.
- [x] `text/encoders/plamo2/sm.hpp` + `text/encoders/plamo2/guards.hpp`
  reworked table-sync, decode-result, and final encode-result routing into
  explicit error-class branches (`*_ok`, `*_invalid_argument_error`,
  `*_backend_error`, `*_model_invalid_error`, `*_unknown_error`) so
  `plamo2/sm.hpp` no longer relies on generic `phase_ok` / `phase_failed`
  guards for action/detail-driven outcomes.
- [x] `token/batcher/sm.hpp` + `token/batcher/guards.hpp`
  reworked seq-normalization, positions, output-mask, and output-counting
  phase routing into explicit error-class branches (`*_ok`,
  `*_invalid_request_error`, `*_backend_error`, `*_internal_error`,
  `*_unknown_error`) so `token/batcher/sm.hpp` no longer relies on generic
  `phase_ok` / `phase_failed` guards for action/detail-driven outcomes.
- [x] `text/jinja/parser/lexer/sm.hpp` + `text/jinja/parser/lexer/guards.hpp` +
  `text/jinja/parser/lexer/actions.hpp`
  reworked string-content intake into explicit
  `string_content_policy_decision` routing
  (`string_scan_immediate_termination_or_eof` vs
  `string_scan_requires_content`) and removed residual runtime `if` branching
  and `?:` branching from string escaped-content action helpers.
- [x] `text/encoders/{bpe,spm,rwkv,wpm}/guards.hpp` +
  `text/encoders/guards.hpp` (removed legacy generic `phase_ok` /
  `phase_failed` guard shims so encoder family routing remains exclusively on
  explicit typed error-class guards used by each machine `sm.hpp`)
- [x] `text/jinja/parser/classifier_parser/sm.hpp` +
  `text/jinja/parser/classifier_parser/guards.hpp`
  reworked classification finalization into explicit
  `classification_result_decision` error-class routing
  (`parse_error_none`, `parse_error_invalid_request`,
  `parse_error_parse_failed`, `parse_error_internal_error`,
  `parse_error_untracked`, `parse_error_unknown`) and removed legacy
  classifier `phase_ok` / `phase_failed` guard shims.
- [x] `batch/planner/modes/sequential/sm.hpp` +
  `batch/planner/modes/sequential/actions.hpp`
  reworked sequential result-failure routing so the prior implicit
  generic `planning_failed` branch is now explicitly modeled as
  `planning_progress_stalled` via
  `planning_result_decision -> planning_failed / mark_planning_progress_stalled`.
- [x] `batch/planner/modes/equal/sm.hpp`
  reworked equal-mode result-failure routing so generic
  `planning_failed` branches from `planning_general_result_decision` and
  `planning_fast_result_decision` now explicitly apply
  `mark_planning_progress_stalled`, making final failure classification
  explicit at the model boundary.
- [x] `batch/planner/modes/simple/sm.hpp` +
  `batch/planner/modes/simple/actions.hpp`
  reworked simple-mode result-failure routing so the prior generic
  `planning_failed` branch now explicitly applies
  `mark_planning_progress_stalled`, making final failure classification
  explicit at the model boundary.
- [x] `batch/planner/sm.hpp` + `batch/planner/guards.hpp`
  reworked parent planner mode-result routing so failed child-mode outcomes are
  classified explicitly at mode completion boundaries via
  `planning_failed_with_error` and `planning_failed_without_error` guard
  branches in `sm.hpp`; removed intermediate generic `plan_failed` phase
  routing and dispatches `_error` outcomes explicitly on each failed-mode path.
- [x] `graph/allocator/liveness_pass/sm.hpp` +
  `graph/allocator/liveness_pass/guards.hpp` +
  `graph/allocator/liveness_pass/actions.hpp`
  reworked liveness pass completion routing to explicit prefailed vs
  done/invalid/capacity branches in `sm.hpp`; removed implicit
  `phase_unclassified_failure` fallback guard and modeled prefailed propagation
  explicitly via `phase_prefailed` / `mark_failed_prefailed`.
