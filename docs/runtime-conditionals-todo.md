# Runtime Conditional Removal TODO

Goal: remove runtime `if (...)` control flow from listed action/detail files by moving orchestration
decisions into guards/explicit dispatch phases and keeping helper kernels branch-model compliant.

Status legend: `[ ]` pending, `[-]` in progress, `[x]` done.

## Actions

- [x] `src/emel/batch/planner/modes/equal/actions.hpp` (0)
- [x] `src/emel/batch/planner/modes/sequential/actions.hpp` (0)
- [x] `src/emel/batch/planner/modes/simple/actions.hpp` (0)
- [x] `src/emel/gbnf/rule_parser/actions.hpp` (0)
- [x] `src/emel/gbnf/rule_parser/lexer/actions.hpp` (0)
- [x] `src/emel/gbnf/sampler/actions.hpp` (0)
- [x] `src/emel/text/renderer/actions.hpp` (0)

## Detail Helpers/Kernels

- [x] `src/emel/batch/planner/modes/detail.hpp` (0)
- [x] `src/emel/docs/detail.hpp` (0)
- [x] `src/emel/gbnf/detail.hpp` (0)
- [x] `src/emel/gbnf/rule_parser/detail.hpp` (0)
- [x] `src/emel/kernel/aarch64/detail.hpp` (0)
- [x] `src/emel/kernel/detail.hpp` (0)
- [x] `src/emel/kernel/x86_64/detail.hpp` (0)
- [x] `src/emel/text/detokenizer/detail.hpp` (0)
- [x] `src/emel/text/encoders/detail.hpp` (0)
- [x] `src/emel/text/jinja/lexer/detail.hpp` (0)
- [x] `src/emel/text/tokenizer/preprocessor/detail.hpp` (0)
- [x] `src/emel/token/batcher/detail.hpp` (0)

## Execution Plan

1. Remove runtime branching from action files first while preserving behavior.
2. Refactor each detail helper file to branch-model-compliant control constructs.
3. Re-run grep-based compliance checks after each batch.
4. Run `scripts/quality_gates.sh`.
