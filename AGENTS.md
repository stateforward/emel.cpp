# AGENTS.md

these rules define the engineering contract for emel.cpp. they are aligned with
`docs/rules/sml.rules.md`. if a rule here conflicts with `docs/rules/sml.rules.md`, the doc
wins and this file must be updated.

## boost.SML actor model (aligned with docs/sml.rules.md)
ALWAYS follow the RTC actor model and no-queue invariant from `docs/sml.rules.md`.
NEVER use `sml::process_queue`, `sml::defer_queue`, or any mailbox/post-for-later
mechanism.
ALWAYS keep dispatch run-to-completion and single-writer per actor.
NEVER call an actor's own `process_event` from guards/actions/entry/exit.
ALWAYS model internal multi-step flows with `sml::completion<TEvent>`,
anonymous transitions, and/or entry actions.
ALWAYS keep anonymous transition graphs acyclic or statically bounded.
ALWAYS propagate originating event data across internal phases via typed
completion transitions (`sml::completion<TEvent>`) when needed.
NEVER use `sml::completion<TEvent>` or anonymous transitions as data-plane
iteration loops (for example per-logit, per-token, or per-tensor-element scans).
ALWAYS keep completion/anonymous chains phase-level only with a small,
statically bounded number of transitions per top-level dispatch.
ALWAYS implement bulk numeric iteration in allocation-free action/detail kernels
within a single transition per phase.
NEVER copy event payload into context just to bridge internal phases.
ALWAYS keep guards pure predicates of `(event, context)` with no side effects.
ALWAYS keep actions bounded and non-blocking during dispatch.
ALWAYS keep hot-path actions allocation-free.
ALWAYS keep any allowed one-time construction or initialization heap
allocation before any `process_event(...)` dispatch.
NEVER perform dynamic allocation during dispatch, including in guards,
actions, entry/exit handlers, or anonymous/completion progress.
ALWAYS model runtime behavior choice as explicit guards or explicit choice
states/transitions.
NEVER hide runtime behavior selection in `actions`, state machine member
functions, or functions called from them.
Treat "runtime behavior selection" semantically, not syntactically.
It includes any helper-local branching, table lookup, flag dispatch, dtype
dispatch, backend dispatch, modality dispatch, model-family dispatch,
fallback selection, block-kind selection, activation-mode selection,
skip/residual selection, buffer-lane selection, callback/error-channel
selection, or other runtime choice that changes which algorithm, path,
variant, or externally observable behavior executes.
NEVER put runtime branching statements (`if`, `else if`, `switch`, `?:`) in
actions, state machine member functions, or functions called from them.
NEVER put validation-path branching in actions or in functions called from
actions; model validation outcomes with guards and explicit transitions.
NEVER emulate runtime branching with loop constructs in actions, detail helpers,
state machine member methods, or functions called from them.
NEVER use single-pass loop patterns such as
`for (bool cond = ...; cond; cond = false)` to choose control paths.
NEVER use branch-case loop patterns such as
`for (size_t emel_case_* = emel_branch_*; ...)` to choose control paths.
NEVER use runtime-indexed handler/candidate arrays (including function-pointer
tables) as a substitute for explicit guards/states/transitions.
ALWAYS use loops in actions/detail only for data-plane iteration with monotonic
progress and bounded work.
Compile-time conditionals (`if constexpr`, `#if`) are allowed inside actions,
state machine member methods, or functions called from actions.
NEVER perform I/O waits, mutex waits, or sleeps inside guards/actions.
ALWAYS inject time via event payloads; NEVER read wall-clock time in guards or
actions.
ALWAYS keep publicly exposed events immutable and small; prefer trivially
copyable payloads when copies are cheap and not on a hot path.
INTERNAL-only events that are not publicly exposed MAY use mutable
pointers/references for synchronous same-RTC handoff.
NEVER expose mutable internal-event payload via public API types.
NEVER retain mutable internal-event payload beyond the top-level dispatch call.
NEVER put owning pointers or dynamic containers in events unless you can prove
no allocation during dispatch.
ALWAYS validate runtime event IDs before using `sml::utility::make_dispatch_table`.
ALWAYS inject a component-local context via SML constructor dependency injection.
NEVER rely on constructor parameter order; prefer a single context aggregate.
ALWAYS use `visit_current_states` or `is(...)` for state inspection.
NEVER allow re-entrancy into the same actor within one RTC chain.
ALWAYS keep cross-actor synchronous calls acyclic and deterministically ordered.
callbacks (`emel::callback`) are allowed only for immediate synchronous replies within the
same RTC chain; invoke before dispatch returns, never store in context, and never call
`process_event` inside the callback.
ALWAYS define explicit behavior for unexpected events; NEVER drop them silently.
ALWAYS use `sml::unexpected_event` for unexpected-event handling (never `event<sml::_>`).
`sml::unexpected_event<_>` is only raised for unhandled external events; it already excludes
internal events. do NOT guard it to exclude `boost::sml::back::internal_event`, or you will
suppress the unexpected event itself.
ALWAYS reproduce a reported bug by adding a failing unit test before making fixes.
ALWAYS keep tracing deterministic, bounded, and allocation-free; use
`sml::logger<...>` when needed.

## architecture and composition
ALWAYS use boost.SML for orchestration state machines.
ALWAYS define transition tables in `struct model` and expose `using sm =
boost::sml::sm<model>;`.
ALWAYS write transition rows in destination-first form:
`sml::state<dst> <= src + event [guard] / action`.
NEVER write source-first rows in new or modified code
(`src + event [guard] / action = dst`).
ALWAYS keep the destination state and `<=` on the same line.
ALWAYS organize large transition tables into explicit visual sections using
phase labels and divider comments (for example
`//------------------------------------------------------------------------------//`).
ALWAYS use leading-comma row style after the first row inside
`make_transition_table(...)`.
ALWAYS use narrowly scoped `// clang-format off/on` around transition tables
when formatting would reduce readability.
ALWAYS use this canonical transition-table skeleton for readability and
consistency:
`return make_transition_table(` then one first row, then each additional row
prefixed with `,`, with destination-first `<=` rows only.
NEVER use macros in models.
ALWAYS keep canonical machine types in component namespaces as `emel::<component>::sm`.
ALWAYS provide additive top-level PascalCase aliases (e.g. `emel::Model`).
ALWAYS map directory layout to namespaces.
ALWAYS keep component files limited to the canonical bases `any`, `context`,
`actions`, `guards`, `errors`, `events`, `sm`, or `detail`, with either `.hpp` or `.cpp`
extensions as needed.
ALWAYS structure machine docs and code namespaces using the pattern
`<domain>/<component>/<type>` for variant families (for example
`memory/coordinator/kv`, `text/tokenizer/preprocessor/bpe`).
ALWAYS colocate machine definition, data, guards, actions, and events within the
same component directory.
NEVER place orchestration logic in data-only files.
ALWAYS put runtime behavior choice in `sm.hpp` transitions using guards from
`guards.hpp`.
NEVER put runtime behavior choice in `actions.hpp`, `detail.hpp`, or `detail.cpp`.
ALWAYS treat `guards.hpp` as the home for runtime predicates that decide which
transition or behavior path is taken.
ALWAYS treat `actions.hpp` as the home for bounded execution of an already-chosen
behavior path.
ALWAYS use `state_`, `event_`, `guard_`, `effect_`, `enter_`, and `exit_`
prefixes for newly introduced state-machine symbols in new or modified code.
ALWAYS treat "symbols" here as transition-table aliases and helper identifiers
such as local state/event aliases, guard/effect functors or functions, and
entry/exit action names.
NEVER rename existing repository state or event type names solely to satisfy
this convention; apply it forward to new symbols you introduce or when
touching code already being refactored.
ALWAYS treat `detail.hpp` and `detail.cpp` as the home for shared hidden private
non-control-flow helpers only.
ONLY put a helper in `detail.hpp` or `detail.cpp` when it is used more than once.
If logic is used by only one owning function or one actor, inline it into that
owner. A helper used only once belongs in the owning `guards.hpp` or
`actions.hpp`, not in `detail.hpp` or `detail.cpp`.
Helpers called from `effect_*`, `enter_*`, `exit_*`, or detail code MUST NOT
choose behavior, route fallback, mode selection, success/error outcome, or
which path runs next.
Those decisions belong only in `guard_*` predicates and `sm.hpp` transitions.
If a helper inspects runtime dtype, backend support, architecture/model name,
tensor layout, modality kind, block flags (for example `has_skip`,
`has_expand`, `has_dw_mid`, `has_se`), activation kind, or scratch-buffer
lane/destination to choose which computation or behavior runs, that helper is
choosing behavior and MUST NOT live in `detail.hpp` or `detail.cpp`.
NEVER put runtime support probing, route fallback, block-kind selection, or
other behavior-selection control flow in `detail.hpp` or `detail.cpp`; model it
in `guards.hpp` and `sm.hpp`.
ALWAYS keep `detail.hpp` and `detail.cpp` helpers non-routing and
non-orchestrating.
ALWAYS keep detail helpers called from actions or state machine member methods
limited to compile-time conditionals and data-plane iteration.
Compile-time conditionals and data-plane iteration are allowed in `detail.hpp`
and `detail.cpp`.
Data-plane branching inside `detail.hpp` / `detail.cpp` is allowed only for the
already-chosen algorithm's local numeric work, bounds handling, padding,
clamping, or monotonic loop progress. It is NOT allowed to select the
algorithm, path, variant, or behavior family itself.
ALWAYS treat "what happens next" as orchestration behavior for `detail.hpp` and
`detail.cpp` review.
Helper output includes return values, out-parameters, reference mutation,
pointer mutation, callback selection, and written context fields.
NEVER use any `detail.hpp` or `detail.cpp` helper output to decide what happens
next.
If a helper affects which guard passes, which action runs, which callback
fires, which event/state/error path is taken, or whether a dispatch is
accepted, rejected, done, or failed, that helper MUST live in `guards.hpp`,
`actions.hpp`, or `sm.hpp`, not in `detail.hpp` or `detail.cpp`.
Superficial cleanup does NOT satisfy these rules. Removing `if` statements from
`actions.hpp` while moving the same runtime choice into `detail.hpp`, helper
return values, helper-selected buffer lanes, or helper-selected variants is
still a rule violation.
Shared non-control-flow helpers in `detail.hpp` or `detail.cpp` MUST use
truthful non-routing verb prefixes such as `compute_`, `validate_`, `bind_`,
`scan_`, `append_`, or `reset_`. Use `compute_` only for data-plane or numeric
work, never as a generic prefix for all helpers. NEVER use routing or
selection verbs for non-guard helpers.
ALWAYS give each machine its own `process_event` wrapper and context ownership.
SHARE behavior via `actions.hpp`/`detail.hpp` helpers or `sm_any`, not inheritance.
ALWAYS keep child-machine data owned by the parent-machine data when composing
machines, and inject parent context into children by reference.
ALWAYS communicate between machines through events and explicit interfaces only.
NEVER call another machine's actions, guards, or functions directly.
NEVER mutate another machine's context directly.
ALWAYS dispatch cross-machine events only via `machine->process_event(...)`.
ALWAYS keep operator arithmetic, lowering, packing, quant/dequant, and
backend-specific numeric work in the owning kernel layer (`src/emel/kernel/**`
or a component-local kernel module when that component explicitly owns
execution).
ALWAYS keep higher layers limited to orchestration, metadata shaping, buffer
binding, and dispatch into kernels.
NEVER add ad hoc operator implementations, backend-specialized loops, lowering
code, or packing helpers throughout generators, planners, loaders, graph
orchestration, wrappers, tests, or tools just to bridge a missing kernel path.
If existing code contains ad hoc compute outside kernel ownership, ALWAYS
migrate that work into the owning kernel surface instead of extending the ad hoc
path.
ALWAYS make plans land new or changed ops in kernel-owned files first;
non-kernel plan tasks may only wire, dispatch, validate, benchmark, or prove
those kernels.
NEVER duplicate code if it's used more than once; put it somewhere it can be shared.

## events, outcomes, and errors
ALWAYS define trigger intent events in the `event` namespace using noun-like,
domain-action names without `cmd_` prefixes.
ALWAYS define machine outcome events in the `events` namespace with explicit
`_done` and `_error` suffixes.
INTERNAL-only `_done`/`_error` events MAY carry mutable payload references when
they are not publicly exposed outside the component boundary.
ALWAYS use references for required event fields.
NEVER model ordinary required event fields as pointers.
ONLY use event payload pointers for optional/nullable fields or C ABI boundary
types that cannot use references.
ONLY allow optional `_done`/`_error` request back-pointers used only for
same-RTC correlation to be nullable pointers under the optional-field rule.
NEVER use `cmd_*`-prefixed event names.
ALWAYS model failures via explicit error states and `_error` events.
NEVER add synthetic fault-injection knobs to production events or actions.
NEVER add test-only control fields to `src/` machine/event payloads.
ALWAYS encode retries and one-shot attempts in the transition graph, not in
mutable context flags.
NEVER mirror explicit state/event outcomes into context members.
NEVER add redundant `status_code` fields when errors are modeled by states/events.
NEVER store per-invocation API output pointers in machine context.

## context rules
ALWAYS define a component-local state-machine context type (e.g. `action::context`).
ALWAYS mutate context inside actions or internal transitions when needed.
NEVER mutate context in guards.
NEVER read or write context directly from state machine member functions.
ALWAYS keep context focused on machine-owned runtime data required across internal
phase events.
NEVER store dispatch-local data in context.
Dispatch-local data includes: current request/event pointers or refs, output pointers,
phase flags, step indexes, temporary counts, and transient error/status codes.
NEVER add context fields named or purposed like: `request`, `event`, `phase`, `step`,
`index`, `count`, `err`, `error`, `status`, `last_error`, `*_out`, `*_ptr`.
ALWAYS pass per-dispatch data across internal phases via typed internal events only
(`events::*_done` / `events::*_error` or typed completion payloads), not context.
ALWAYS keep context fields to persistent actor-owned state meaningful across
top-level dispatch calls.
If a machine has no persistent actor-owned state, context MUST be an empty struct.
NEVER mirror per-dispatch request/event payload fields into context only for
phase handoff.
NEVER store orchestration phase/attempt/failure flags in context.
NEVER add string/pointer `error` members to machine data.

## API boundaries
ALWAYS use `extern \"C\"` for public API function signatures.
ALWAYS use fixed-width integer types at API boundaries.
ALWAYS return error codes across API boundaries.
NEVER throw exceptions across API boundaries.
NEVER expose C++ templates, classes, or STL containers directly in the public C
ABI.

## performance and allocation
ALWAYS treat performance as a top-level priority.
NEVER use dynamic dispatch in inference hot paths unless explicitly justified.
ALWAYS prefer compile-time polymorphism in hot paths.
NEVER allocate in inference or sampling hot paths.
NEVER use heap allocation by default.
ALWAYS allow one-time heap allocation during construction, initialization, or
other non-hot-path setup work when necessary.
ALWAYS reuse unavoidable heap allocations.
ALWAYS document rationale for unavoidable heap allocation in code.
ALWAYS keep telemetry non-blocking and optional.
NEVER use exceptions for control flow in hot paths.
NEVER rely on try/catch inside state machine actions/guards; avoid exceptions
unless absolutely necessary.
ALWAYS keep actor models independent; do not share a model between actors unless
explicitly authorized by the user. Only common actions/guards that are not
variant-specific may be shared.

## naming, style, and portability
ALWAYS use snake_case for functions, variables, and namespaces.
ALWAYS use PascalCase for exported/public C++ types only (not C API types).
ALWAYS use lower_snake_case for non-exported/internal types, SML state names, and events.
ALWAYS use SCREAMING_SNAKE_CASE for constants and macros.
ALWAYS keep line length near 100 columns and use 2-space indentation.
NEVER use `using namespace` in headers.
ALWAYS keep code portable across linux, macOS, and windows.
NEVER use platform-specific APIs unless wrapped behind an abstraction.
ALWAYS write scripts to work on any Unix-based OS.

## build, tests, and CI gates
ALWAYS work on a feature branch and submit changes via a pull request.
NEVER push directly to `main`.
NEVER commit `tmp/llama.cpp`.
ALWAYS use zig toolchain (zig cc and zig c++) for default development and
production builds.
ALWAYS use native clang or gcc for coverage builds.
ALWAYS use doctest for unit tests.
ALWAYS use SML introspection for machine assertions and testing.
ALWAYS name test files by machine or domain (e.g. `tests/inference/sampler_tests.cpp`).
NEVER use arbitrary or ad-hoc test file names.
NEVER create monolithic test files.
ALWAYS scope each test file to one machine, one system, or one behavior.
ALWAYS keep one machine benchmark per `tools/bench` source file.
ALWAYS keep snapshot baselines under `snapshots/` and lint baselines under
`snapshots/lint/`.
NEVER update snapshots without explicit user consent.
ALWAYS hard-fail runs when required tools are missing.
ALWAYS enforce line coverage >= 90%.
ALWAYS run `scripts/quality_gates.sh` after each implementation change.
ALWAYS use ctest targets `emel_tests` and `lint_snapshot` for test execution.
ALWAYS reference `docs/rules/sml.rules.md` for SML semantics and testing guidance.

## reference policy
ALWAYS treat `src/` boost.SML machines as the single source of truth for
architecture and orchestration.
ALWAYS use `src/emel/gbnf` as the default architectural reference for new
Boost.SML machine organization, decomposition, and transition-table layout,
unless the current task explicitly requires a different reference family.
NEVER maintain parallel machine-definition markdown specs under
`docs/architecture/*`.
ALWAYS document state purpose, key invariants, guard semantics, and action side
effects.
ALWAYS treat the reference implementation as the functional logic reference for
allocator and behavioral parity work.
NEVER port reference control flow, branching structure, lifecycle semantics, or
orchestration decisions verbatim from llama.cpp/ggml.
ALWAYS define EMEL behavior and orchestration semantics in boost.SML machines as
source of truth.
ALWAYS port llama.cpp/ggml arithmetic, kernels, and instruction behavior into
this codebase when implementing equivalent EMEL functionality.
ALWAYS preserve or improve performance when porting logic from the reference
implementation.
ALWAYS implement equivalent functionality natively without external llama.cpp or
ggml linkage.
NEVER link "emel" against llama.cpp or ggml outside `tools/bench` or
`tools/paritychecker`.
ALWAYS keep any llama.cpp/ggml linkage in `tools/bench` and
`tools/paritychecker` confined to the explicit reference-side comparison path
only.
NEVER let the EMEL side of `tools/bench` or `tools/paritychecker` call into,
bootstrap from, or otherwise depend on llama.cpp/ggml for vocab state,
tokenizer state, formatter state, model loading, tensor metadata, runtime
execution, or output generation.
NEVER share llama.cpp/ggml-created model, vocab, tokenizer, formatter, context,
or cache objects with the EMEL side in benchmark or parity harnesses.
ALWAYS keep benchmark and parity harnesses split into two clearly separated
lanes: an EMEL-owned lane using only EMEL-owned code for the EMEL result, and a
reference lane using llama.cpp/ggml only for the comparison result.
NEVER let parity harnesses or benchmarks reach into actor `actions.hpp`,
`detail.hpp`, or `detail.cpp` helpers directly; drive them through the owning
state machines via `process_event(...)` and public event interfaces only.
NEVER use `llama_` or `ggml_` prefixes in identifiers, symbols, files, or APIs
outside `tools/bench` or `tools/paritychecker`.
ALWAYS use `emel_` or `EMEL_` prefixes for project-owned identifiers, symbols,
files, and APIs.
ALWAYS prefer porting the exact reference arithmetic and kernel operand path
into EMEL-owned `src/` code when the user asks for parity, benchmarking, or
performance comparison against llama.cpp or ggml.
ALWAYS treat reference-kernel parity work as incomplete until the corresponding
EMEL-owned kernel consumes the same effective operand format as the reference
path.
NEVER replace a missing native packed or quantized kernel in a hot inference
path with a dequantize-to-f32 fallback unless the user explicitly approves that
fallback as an interim milestone.
If an interim fallback is explicitly approved, ALWAYS label it `interim` in
code comments, tests, milestone docs, and user updates, and ALWAYS state which
exact reference operand or kernel path is still missing.
NEVER present parity or benchmark results as kernel parity when EMEL and the
reference implementation are not executing materially equivalent operand
pipelines.
ALWAYS ask the user before landing an implementation that changes the
performance contract of a milestone by substituting a simpler fallback kernel,
helper backend, or tool-local scaffold for the intended `src/` runtime path.
For quantized inference work, "done" means:
1. EMEL-owned code in `src/`
2. no tool-only compute fallback
3. no whole-tensor dequantize-to-f32 substitution in the hot path
4. benchmark claims based on the same effective operand class as the reference
   path
When an implementation is architecturally narrower than the user's stated
goal, ALWAYS stop and get explicit approval before proceeding, even if the
narrower implementation is faster to complete.
