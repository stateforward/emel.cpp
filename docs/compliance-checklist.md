# Architecture Compliance Checklist (Hard Requirements)

This checklist is architecture-only and merge-blocking for machine design/orchestration.

- Primary authority: `docs/rules/sml.rules.md`.
- Contract mirror: `AGENTS.md`.
- If these conflict, `docs/rules/sml.rules.md` wins.
- Any unchecked item is a compliance failure and blocks merge.

## 1) SML Actor Architecture

- [ ] Orchestration uses Boost.SML state machines (`boost::sml::sm<...>`); no alternate orchestration framework.
- [ ] Each machine defines transitions in `struct model` and exposes a canonical machine type (`struct sm : emel::sm<model, ...>` or an equivalent local alias pattern).
- [ ] Transition rows use destination-first form only: `sml::state<dst> <= src + event [guard] / action`.
- [ ] No source-first transition syntax is introduced in new/modified code.
- [ ] Transition tables use canonical layout: first row, then leading commas for subsequent rows.
- [ ] Large transition tables are visually sectioned with explicit phase labels/dividers.
- [ ] No SML queue policies (`sml::process_queue`, `sml::defer_queue`) and no mailbox/post-for-later mechanism.
- [ ] Dispatch remains run-to-completion (RTC) and single-writer per actor.
- [ ] Actor self-reentrancy is forbidden (`process_event` is never called on self from guards/actions/entry/exit).
- [ ] Internal multi-phase flow uses typed completion, anonymous transitions, and/or entry actions (not self-dispatch).
- [ ] Anonymous/completion chains are acyclic or statically bounded per top-level dispatch.
- [ ] Completion/anonymous transitions are never used for data-plane loops (per-token/logit/tensor-element scans).
- [ ] Bulk numeric loops execute inside allocation-free action/detail kernels within a single phase transition.
- [ ] Cross-machine interaction happens only through events and `machine->process_event(...)`.
- [ ] Machines never call other machines’ actions/guards/member internals directly.
- [ ] Machines never mutate another machine’s context directly.
- [ ] Parent machines own child-machine data; children receive parent context by reference.
- [ ] Each machine has its own `process_event` wrapper and context ownership.
- [ ] Directory layout maps to namespaces; canonical machine type remains `emel::<component>::sm`.
- [ ] New machine/component files are limited to: `any.hpp`, `context.hpp`, `actions.hpp`, `guards.hpp`, `errors.hpp`, `sm.hpp`, `detail.hpp`.

## 2) Action and Guard Architecture

- [ ] Guards are pure predicates of `(event, context)` and have no side effects.
- [ ] Guards never mutate context.
- [ ] Actions are bounded and non-blocking.
- [ ] Runtime branching statements (`if`, `else if`, `switch`, `?:`) are not implemented inside actions/member methods.
- [ ] Runtime branching statements (`if`, `else if`, `switch`, `?:`) are not implemented in functions called from actions/member methods.
- [ ] Runtime control flow is modeled only as explicit guarded transitions or explicit choice states.
- [ ] Runtime branch emulation via single-pass conditional loops is absent in `actions.hpp`/`detail.hpp` (`for (bool cond = ...; cond; cond = false)`).
- [ ] Runtime branch emulation via branch-case loops is absent in `actions.hpp`/`detail.hpp` (`for (size_t emel_case_* = emel_branch_*; ...)`).
- [ ] Runtime-indexed handler/candidate dispatch selection is not used in actions/detail as a control-flow substitute (allowed only for data lookup).
- [ ] Loops in actions/detail are data-plane iteration only (monotonic progress, bounded work), not success/error/mode/retry/routing control.
- [ ] Only compile-time conditionals (`if constexpr`, `#if`) appear in actions/member methods/action callees.
- [ ] Anti-shortcut lint gate (or no-new-violations ratchet) passes and is attached to the PR.
- [ ] State-machine member functions do not read/write context directly.

## 3) Event, Error, and Context Architecture

- [ ] Trigger intent events are in `event` namespace with noun-like domain-action names; no `cmd_*`.
- [ ] Outcome events are in `events` namespace and use explicit `_done` / `_error` suffixes.
- [ ] Failures are modeled by explicit error states/events; not by context status flags.
- [ ] Required event fields are references (never pointers).
- [ ] Event pointers are used only for optional/nullable fields or C ABI boundary constraints.
- [ ] Public events are immutable, small, and preferably trivially copyable.
- [ ] Internal-only mutable payload in events is not exposed in public API types.
- [ ] Internal mutable event payload is not retained beyond the top-level dispatch call.
- [ ] Event payload avoids owning pointers/dynamic containers unless no-allocation during dispatch is proven.
- [ ] Runtime event IDs are range-validated before `sml::utility::make_dispatch_table` indexing.
- [ ] Context is component-local and stores only persistent actor-owned state across top-level dispatches.
- [ ] Context is never used for dispatch-local scratch (request/event mirrors, phase/step/index/count, transient status/error).
- [ ] Context never includes per-invocation output pointers or string/pointer `error` members.
- [ ] No global/shared error enum is used as orchestration control state.
- [ ] Error typing is component-local (in machine-local `errors.hpp`) and propagated via explicit `_error` events/states.
- [ ] Per-dispatch phase handoff uses typed internal events (`*_done`, `*_error`, `sml::completion<TEvent>`), not context mirroring.
- [ ] Unexpected external events are handled explicitly with `sml::unexpected_event`.
- [ ] `event<sml::_>` is not used for unexpected-event handling.

## 4) Pattern and Convention Enforcement (Kernel, GBNF, Memory)

- [ ] `src/` Boost.SML machines are the source of truth for orchestration/architecture.
- [ ] `src/emel/gbnf`, `src/emel/kernel`, and `src/emel/memory` patterns are enforced as architecture references where applicable.
- [ ] `src/emel/gbnf` remains the default structural reference family when task scope does not require a different family.
- [ ] No parallel machine-definition specs are introduced under `docs/architecture/*`.
- [ ] EMEL orchestration semantics are defined natively in EMEL SML machines (no verbatim external control-flow port).
- [ ] Parent/child composition follows the memory convention: parent owns child machine data and injects child context by reference.
- [ ] Cross-machine orchestration follows the family convention: interactions happen only through explicit events and `process_event(...)`.
- [ ] Wrapper convention is enforced: public request events are adapted into internal runtime events with stack-local per-dispatch runtime context.
- [ ] Wrapper convention is enforced: success is derived from machine acceptance plus runtime error context, without persisting dispatch-local state.
- [ ] Optional output handling follows wrapper conventions (bind-or-sink style), never by storing per-dispatch output pointers in machine context.
- [ ] Backend routing follows kernel conventions: explicit backend actor fanout, explicit per-backend acceptance/outcome fields, and deterministic documented order.
- [ ] Backend endpoint conventions are consistent: each backend machine exposes a compatible typed `process_event` surface.
- [ ] Parser leaf-machine conventions follow GBNF: classifier-focused leaf states, explicit success/failure terminals, and explicit `sml::X` exits.
- [ ] Multi-phase orchestrator conventions follow GBNF/memory: explicit request/decision/execute/result phase states and explicit completion/error states.
- [ ] State naming conventions follow family patterns (`*_decision`, `*_exec`, `*_result_decision`, `done`, `errored`, and domain-specific variants when needed).
- [ ] Unexpected-event convention is enforced: explicit `sml::unexpected_event` handling from each relevant state with deterministic recovery/termination.
- [ ] Error typing conventions are local to each component family (`errors.hpp` per component); no global/shared orchestration error enum.
- [ ] Public machine namespace conventions include additive PascalCase aliases for canonical machine types.

## 5) Architecture Sign-Off

- [ ] State/transition architecture review passed.
- [ ] Event/error/context architecture review passed.
- [ ] Action/guard architecture review passed.
- [ ] Kernel/GBNF/Memory reference-pattern review passed.
