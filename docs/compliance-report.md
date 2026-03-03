# Compliance Report

Generated: 2026-03-02 23:24:38 CST

Scope:
- Audited every machine definition under `src/emel/**/sm.hpp` (81 machine files).
- Excluded framework wrapper `src/emel/sm.hpp` from machine scoring.

Method:
- Static analysis only (regex/structural checks over source tree).
- Checklist source: [docs/compliance-checklist.md](docs/compliance-checklist.md).
- Static checks are exhaustive for all machine files; non-static items are marked `MANUAL`.

## Snapshot

- Machines audited: **81**
- State-table machines (contain `make_transition_table(...)`): **79**
- Static core pass/fail: **78 / 3**
- Source-first transition syntax offenders: **1**
- Machines with explicit `sml::unexpected_event<...>`: **79 / 79**
- Queue/mailbox policy usage: **0**
- `sml::event<sml::_>` usage: **0**
- Machines with canonical `sm` type (struct or alias): **81 / 81**
- Machines with public `process_event` wrapper: **45 / 81**
- State-table machines without public `process_event` wrapper: **34**
- Benchmark marker distribution: scaffold **37**, ready **0**, none **44**
- Actions files with runtime `if (...)`: **7**
- Detail files with runtime `if (...)`: **12**
- Guards files with direct `ctx.*` mutation patterns: **0**
- Actions/detail files containing `process_event(...)` cross-machine dispatch: **10**
- Machine dirs violating filename whitelist item: **67**
- Machine files with PascalCase alias (e.g. `using Foo = sm;`): **27 / 81**

## Checklist Mapping

Status legend: `PASS` = met by static evidence, `FAIL` = violated by static evidence, `PARTIAL` = mixed static evidence, `MANUAL` = requires human semantic review, `N/A` = check not applicable in current code path.

### 1) SML Actor Architecture

| Item | Status | Evidence |
| --- | --- | --- |
| 1.1 Boost.SML orchestration | PASS | 79 machine files contain `make_transition_table(...)`; remaining wrappers alias to machine types |
| 1.2 `struct model` + canonical `sm` type | PARTIAL | 79 files have `struct model`; 81 have `sm` type; alias-only wrappers: `src/emel/kernel/sm.hpp`, `src/emel/text/encoders/sm.hpp` |
| 1.3 Destination-first transition rows only | FAIL | source-first rows in `src/emel/text/formatter/sm.hpp:65`, `src/emel/text/formatter/sm.hpp:66` |
| 1.4 No source-first syntax in modified code | FAIL | same offender as 1.3 |
| 1.5 Canonical table layout + leading commas | PARTIAL | 78 files show divider+leading-comma row style; `text/formatter/sm.hpp` does not |
| 1.6 Large tables visually sectioned | PARTIAL | 78/81 machine files include divider blocks |
| 1.7 No `process_queue` / `defer_queue` / mailbox | PASS | zero matches repo-wide in machine files |
| 1.8 RTC single-writer per actor | MANUAL | semantic/runtime property |
| 1.9 No self re-entrancy (`process_event` on self in actions/guards) | PASS | no self-call patterns in actions/guards/detail |
| 1.10 Internal multi-phase uses completion/anonymous/entry | MANUAL | semantic flow check |
| 1.11 Anonymous/completion chains bounded/acyclic | MANUAL | semantic graph check |
| 1.12 No completion/anonymous data-plane loops | MANUAL | semantic intent check |
| 1.13 Bulk loops in bounded kernels | MANUAL | requires per-action review |
| 1.14 Cross-machine interaction only via events + `process_event` | PASS | all observed cross-machine calls use `process_event(...)` in actions/detail |
| 1.15 No direct calls to other machine internals | PASS | no direct action/guard/member internal calls found |
| 1.16 No mutation of another machine context | MANUAL | semantic ownership check |
| 1.17 Parent owns child data / child gets parent context by ref | MANUAL | composition review required |
| 1.18 Each machine has own `process_event` wrapper + context ownership | FAIL | 34 state-table submachines lack public `process_event` wrapper |
| 1.19 Directory layout maps namespaces + canonical machine type | PARTIAL | generally aligned; alias-only wrapper files exist |
| 1.20 File whitelist (`any/context/actions/guards/errors/sm/detail`) | FAIL | 67 machine dirs include non-whitelist files (mostly `events.hpp`) |

### 2) Action and Guard Architecture

| Item | Status | Evidence |
| --- | --- | --- |
| 2.1 Guards are pure predicates | PASS | no guard context mutation patterns detected |
| 2.2 Guards never mutate context | PASS | `guards_with_ctx_mutation = 0` |
| 2.3 Actions bounded / non-blocking | MANUAL | semantic/runtime property |
| 2.4 No runtime control-flow in actions | FAIL | runtime `if (...)` in 7 `actions.hpp` files |
| 2.5 No runtime control-flow in action callees | FAIL | runtime `if (...)` in 12 `detail.hpp` files |
| 2.6 Runtime flow modeled as guards/choice states | FAIL | contradicted by 2.4/2.5 |
| 2.7 Only compile-time conditionals in actions/callees | FAIL | contradicted by 2.4/2.5 |
| 2.8 SM member functions do not read/write context directly | PASS | wrappers call base `process_event` with runtime event/context, no direct field mutation detected |

### 3) Event, Error, and Context Architecture

| Item | Status | Evidence |
| --- | --- | --- |
| 3.1 Trigger events in `event` namespace, no `cmd_*` | PASS | zero `cmd_` matches in `src/emel`/`include/emel` |
| 3.2 Outcome events in `events` namespace with `_done/_error` | MANUAL | naming audit requires semantic classification of outcomes vs internal structs |
| 3.3 Failures modeled via explicit states/events | PARTIAL | many machines do; full semantic verification required |
| 3.4 Required fields as references (not pointers) | MANUAL | requires per-event API contract review |
| 3.5 Pointer fields only optional/ABI-constrained | MANUAL | semantic field-role review required |
| 3.6 Public events immutable/small | MANUAL | API-level review required |
| 3.7 Internal mutable payload not exposed publicly | MANUAL | boundary review required |
| 3.8 Internal mutable payload not retained beyond dispatch | MANUAL | lifetime review required |
| 3.9 No owning pointers/dynamic containers in events unless proven | MANUAL | event payload review required |
| 3.10 Event ID validation before `make_dispatch_table` indexing | N/A | no `make_dispatch_table` usage found |
| 3.11 Context is component-local persistent state | PARTIAL | contexts are local, persistence semantics vary by component |
| 3.12 Context not used for dispatch-local scratch | MANUAL | requires per-context semantic review |
| 3.13 Context avoids per-invocation output/error pointer members | MANUAL | requires per-context field intent review |
| 3.14 No global/shared orchestration error enum | PASS | no shared global enum used as orchestration control state |
| 3.15 Error typing component-local (`errors.hpp`) | PARTIAL | 50/81 machine dirs have local `errors.hpp`; some leaf dirs rely on parent error types |
| 3.16 Dispatch handoff via typed internal events, not context mirroring | MANUAL | semantic event-flow review required |
| 3.17 Unexpected external events explicitly handled via `sml::unexpected_event` | PARTIAL | 79/79 state-table machines handle unexpected events; alias-only wrappers have no transition table |
| 3.18 `event<sml::_>` not used for unexpected handling | PASS | zero `sml::event<sml::_>` matches |

### 4) Pattern and Convention Enforcement (Kernel/GBNF/Memory)

| Item | Status | Evidence |
| --- | --- | --- |
| 4.1 `src/` SML machines are source of truth | PASS | machine definitions exist under `src/emel/**/sm.hpp` |
| 4.2 Kernel/GBNF/Memory family patterns enforced | MANUAL | architecture-level human review required |
| 4.3 GBNF as default structural reference family | MANUAL | process/policy check |
| 4.4 No parallel machine-definition specs under `docs/architecture/*` | MANUAL | docs are present; intent (generated docs vs parallel specs) requires policy decision |
| 4.5 Native EMEL orchestration semantics | MANUAL | semantic parity review required |
| 4.6 Parent/child memory composition conventions | MANUAL | deep component review required |
| 4.7 Cross-machine orchestration uses events + `process_event` | PASS | static evidence in action/detail cross-calls |
| 4.8 Wrapper convention: request -> stack-local runtime event/context | PARTIAL | present in 46 machine wrappers; leaf submachines omit wrappers |
| 4.9 Wrapper convention: success = acceptance + runtime error context | PARTIAL | widely used in wrappers, not universal across leaf submachines |
| 4.10 Optional-output handling via wrapper conventions | MANUAL | per-wrapper semantic check required |
| 4.11 Kernel backend routing conventions enforced | PARTIAL | explicit backend fanout exists; full determinism check requires manual review |
| 4.12 Backend endpoints expose compatible typed `process_event` surface | PARTIAL | largely true; full type-surface review required |
| 4.13 GBNF parser leaf-machine conventions | PARTIAL | many leaf parsers match pattern; manual confirmation needed |
| 4.14 Multi-phase orchestrator conventions | PARTIAL | many orchestrators follow request/decision phases; manual confirmation needed |
| 4.15 State naming conventions (`*_decision`, `done`, `errored`, etc.) | PARTIAL | broadly followed, not universally enforced by static checks |
| 4.16 Unexpected-event convention from relevant states | PARTIAL | unexpected handlers present in state-table machines; per-state completeness needs manual review |
| 4.17 Local component `errors.hpp` conventions | PARTIAL | see 3.15 |
| 4.18 PascalCase alias conventions for public machine namespaces | FAIL | only 27/81 machine files define a PascalCase alias |

### 5) Architecture Sign-off

| Item | Status | Evidence |
| --- | --- | --- |
| 5.1 State/transition architecture review passed | FAIL | open FAIL/PARTIAL items in section 1 |
| 5.2 Event/error/context architecture review passed | FAIL | open PARTIAL/MANUAL items in section 3 |
| 5.3 Action/guard architecture review passed | FAIL | runtime conditionals in actions/detail |
| 5.4 Kernel/GBNF/Memory reference-pattern review passed | FAIL | open PARTIAL/MANUAL items in section 4 |

## Static Violations (Actionable)

1. Source-first transition syntax still present:
   - `src/emel/text/formatter/sm.hpp:65`
   - `src/emel/text/formatter/sm.hpp:66`
2. Alias-only machine wrappers without local model/transition table (treated as machine files by inventory):
   - `src/emel/kernel/sm.hpp`
   - `src/emel/text/encoders/sm.hpp`
3. Runtime conditionals in actions (`if (...)` not `if constexpr`):
   - `src/emel/batch/planner/modes/equal/actions.hpp`
   - `src/emel/batch/planner/modes/sequential/actions.hpp`
   - `src/emel/batch/planner/modes/simple/actions.hpp`
   - `src/emel/gbnf/rule_parser/actions.hpp`
   - `src/emel/gbnf/rule_parser/lexer/actions.hpp`
   - `src/emel/gbnf/sampler/actions.hpp`
   - `src/emel/text/renderer/actions.hpp`
4. Runtime conditionals in detail kernels/helpers (called from actions):
   - `src/emel/batch/planner/modes/detail.hpp`
   - `src/emel/docs/detail.hpp`
   - `src/emel/gbnf/detail.hpp`
   - `src/emel/gbnf/rule_parser/detail.hpp`
   - `src/emel/kernel/aarch64/detail.hpp`
   - `src/emel/kernel/detail.hpp`
   - `src/emel/kernel/x86_64/detail.hpp`
   - `src/emel/text/detokenizer/detail.hpp`
   - `src/emel/text/encoders/detail.hpp`
   - `src/emel/text/jinja/lexer/detail.hpp`
   - `src/emel/text/tokenizer/preprocessor/detail.hpp`
   - `src/emel/token/batcher/detail.hpp`
5. File whitelist rule mismatch:
   - 67 machine dirs violate the whitelist; most violations are `events.hpp` presence.

## Per-Machine Static Matrix

Scoring rule for `StaticStatus`:
- `PASS`: has model+transition table, canonical `sm` type, zero source-first rows, explicit unexpected-event handling, no queue/mailbox policy usage, no `event<sml::_>` usage, and canonical formatting checks satisfied.
- `FAIL`: one or more of the above conditions violated.

| Machine | StaticStatus | Model | Table | SMType | ProcessWrapper | SourceFirstRows | Unexpected | QueuePolicy | EventSmlAny | Divider | LeadingComma | Benchmark |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `src/emel/batch/planner/modes/equal/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/batch/planner/modes/sequential/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/batch/planner/modes/simple/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/batch/planner/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/gbnf/rule_parser/definition_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/gbnf/rule_parser/expression_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/gbnf/rule_parser/lexer/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/gbnf/rule_parser/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/gbnf/rule_parser/term_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/gbnf/sampler/accept_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/gbnf/sampler/candidate_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/gbnf/sampler/matcher_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/gbnf/sampler/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/gbnf/sampler/token_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/generator/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/gguf/loader/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/graph/allocator/liveness_pass/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/allocator/ordering_pass/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/allocator/placement_pass/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/allocator/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/graph/assembler/assemble_alloc_pass/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/assembler/assemble_build_pass/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/assembler/assemble_validate_pass/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/assembler/reserve_alloc_pass/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/assembler/reserve_build_pass/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/assembler/reserve_validate_pass/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/assembler/reuse_decision_pass/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/assembler/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/graph/processor/alloc_step/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/processor/bind_step/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/processor/extract_step/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/processor/kernel_step/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/processor/prepare_step/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/processor/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/graph/processor/validate_step/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/graph/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/kernel/aarch64/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/kernel/cuda/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/kernel/metal/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/kernel/sm.hpp` | FAIL | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | none |
| `src/emel/kernel/vulkan/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/kernel/wasm/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/kernel/x86_64/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/logits/sampler/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/logits/validator/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/memory/hybrid/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/memory/kv/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/memory/recurrent/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/model/loader/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/model/weight_loader/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/tensor/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/tensor/view/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | scaffold |
| `src/emel/text/conditioner/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/detokenizer/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/encoders/bpe/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/encoders/fallback/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/encoders/plamo2/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/encoders/rwkv/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/encoders/sm.hpp` | FAIL | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | scaffold |
| `src/emel/text/encoders/spm/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/encoders/ugm/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/encoders/wpm/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/formatter/sm.hpp` | FAIL | 1 | 1 | 1 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | scaffold |
| `src/emel/text/jinja/formatter/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/jinja/parser/classifier_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/jinja/parser/lexer/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/jinja/parser/program_parser/expression_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/jinja/parser/program_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/jinja/parser/program_parser/statement_parser/sm.hpp` | PASS | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/jinja/parser/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/renderer/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/tokenizer/preprocessor/bpe/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/tokenizer/preprocessor/fallback/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/tokenizer/preprocessor/plamo2/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/tokenizer/preprocessor/rwkv/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/tokenizer/preprocessor/spm/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/tokenizer/preprocessor/ugm/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/tokenizer/preprocessor/wpm/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/text/tokenizer/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |
| `src/emel/token/batcher/sm.hpp` | PASS | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | none |

## Notes

- This report is static-analysis driven. Any `MANUAL` item requires targeted human architecture review.
- The checklist item limiting component files to `any/context/actions/guards/errors/sm/detail` conflicts with current architecture patterns that widely use `events.hpp`; the report marks this as a hard FAIL per checklist text.
- `src/emel/kernel/sm.hpp` and `src/emel/text/encoders/sm.hpp` are alias wrappers; they are included for completeness because they are `sm.hpp` machine entry files in the tree.
