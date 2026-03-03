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
- [x] `src/emel/text/encoders/fallback/detail.hpp`

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
