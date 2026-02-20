# AGENTS.md

These rules define the engineering contract for emel.cpp. They are aligned with
`docs/sml.rules.md`. If a rule here conflicts with `docs/sml.rules.md`, the doc
wins and this file must be updated.

## Boost.SML Actor Model (Aligned With docs/sml.rules.md)
ALWAYS follow the RTC actor model and no-queue invariant from `docs/sml.rules.md`.
NEVER use `sml::process_queue`, `sml::defer_queue`, or any mailbox/post-for-later
mechanism.
ALWAYS keep dispatch run-to-completion and single-writer per actor.
NEVER call an actor's own `process_event` from guards/actions/entry/exit.
ALWAYS model internal multi-step flows with anonymous transitions and/or entry
actions.
ALWAYS keep anonymous transition graphs acyclic or statically bounded.
ALWAYS keep guards pure predicates of `(event, context)` with no side effects.
ALWAYS keep actions bounded, non-blocking, and allocation-free during dispatch.
NEVER perform I/O waits, mutex waits, or sleeps inside guards/actions.
ALWAYS inject time via event payloads; NEVER read wall-clock time in guards or
actions.
ALWAYS keep events immutable and small; prefer trivially copyable payloads.
NEVER put owning pointers or dynamic containers in events unless you can prove
no allocation during dispatch.
ALWAYS validate runtime event IDs before using `sml::utility::make_dispatch_table`.
ALWAYS inject a component-local context via SML constructor dependency injection.
NEVER rely on constructor parameter order; prefer a single context aggregate.
ALWAYS use `visit_current_states` or `is(...)` for state inspection.
NEVER allow re-entrancy into the same actor within one RTC chain.
ALWAYS keep cross-actor synchronous calls acyclic and deterministically ordered.
Callbacks (`emel::callback`) are allowed only for immediate synchronous replies within the
same RTC chain; invoke before dispatch returns, never store in context, and never call
`process_event` inside the callback.
ALWAYS define explicit behavior for unexpected events; NEVER drop them silently.
ALWAYS use `sml::unexpected_event` for unexpected-event handling (never `event<sml::_>`).
`sml::unexpected_event<_>` is only raised for unhandled external events; it already excludes
internal events. Do NOT guard it to exclude `boost::sml::back::internal_event`, or you will
suppress the unexpected event itself.
ALWAYS reproduce a reported bug by adding a failing unit test before making fixes.
ALWAYS keep tracing deterministic, bounded, and allocation-free; use
`sml::logger<...>` when needed.

## Architecture and Composition
ALWAYS use Boost.SML for orchestration state machines.
ALWAYS define transition tables in `struct model` and expose `using sm =
boost::sml::sm<model>;`.
ALWAYS keep canonical machine types in component namespaces as `emel::<component>::sm`.
ALWAYS provide additive top-level PascalCase aliases (e.g. `emel::Model`).
ALWAYS map directory layout to namespaces.
ALWAYS colocate machine definition, data, guards, actions, and events within the
same component directory.
NEVER place orchestration logic in data-only files.
ALWAYS keep child-machine data owned by the parent-machine data when composing
machines, and inject parent context into children by reference.
ALWAYS communicate between machines through events and explicit interfaces only.
NEVER call another machine's actions, guards, or functions directly.
NEVER mutate another machine's context directly.
ALWAYS dispatch cross-machine events only via `machine->process_event(...)`.
ALWAYS ask the user before changing state machine structure.

## Events, Outcomes, and Errors
ALWAYS define trigger intent events in the `event` namespace using noun-like,
domain-action names without `cmd_` prefixes.
ALWAYS define machine outcome events in the `events` namespace with explicit
`_done` and `_error` suffixes.
NEVER use `cmd_*`-prefixed event names.
ALWAYS model failures via explicit error states and `_error` events.
NEVER add synthetic fault-injection knobs to production events or actions.
NEVER add test-only control fields to `src/` machine/event payloads.
ALWAYS encode retries and one-shot attempts in the transition graph, not in
mutable context flags.
NEVER mirror explicit state/event outcomes into context members.
NEVER add redundant `status_code` fields when errors are modeled by states/events.
NEVER store per-invocation API output pointers in machine context.

## Context Rules
ALWAYS define a component-local state-machine context type (e.g. `action::context`).
ALWAYS mutate context inside actions or internal transitions when needed.
NEVER mutate context in guards.
ALWAYS keep context focused on machine-owned runtime data required across internal
phase events.
NEVER store orchestration phase/attempt/failure flags in context.
NEVER add string/pointer `error` members to machine data.

## API Boundaries
ALWAYS use `extern \"C\"` for public API function signatures.
ALWAYS use fixed-width integer types at API boundaries.
ALWAYS return error codes across API boundaries.
NEVER throw exceptions across API boundaries.
NEVER expose C++ templates, classes, or STL containers directly in the public C
ABI.

## Performance and Allocation
ALWAYS treat performance as a top-level priority.
NEVER use dynamic dispatch in inference hot paths unless explicitly justified.
ALWAYS prefer compile-time polymorphism in hot paths.
NEVER allocate in inference or sampling hot paths.
NEVER use heap allocation by default.
ALWAYS allow one-time heap allocation only at construction or initialization.
ALWAYS reuse unavoidable heap allocations.
ALWAYS document rationale for unavoidable heap allocation in code.
ALWAYS keep telemetry non-blocking and optional.
NEVER use exceptions for control flow in hot paths.

## Naming, Style, and Portability
ALWAYS use snake_case for functions, variables, and namespaces.
ALWAYS use PascalCase for exported/public types and SML state names.
ALWAYS use lower_snake_case for non-exported/internal types (e.g., src-only aliases).
ALWAYS use SCREAMING_SNAKE_CASE for constants and macros.
ALWAYS keep line length near 100 columns and use 2-space indentation.
NEVER use `using namespace` in headers.
ALWAYS keep code portable across Linux, macOS, and Windows.
NEVER use platform-specific APIs unless wrapped behind an abstraction.

## Build, Tests, and CI Gates
ALWAYS use Zig toolchain (zig cc and zig c++) for default development and
production builds.
ALWAYS use native clang or gcc for coverage builds.
ALWAYS use doctest for unit tests.
ALWAYS use SML introspection for machine assertions and testing.
ALWAYS name test files by machine or domain (e.g. `tests/inference/sampler_tests.cpp`).
NEVER use arbitrary or ad-hoc test file names.
NEVER create monolithic test files.
ALWAYS scope each test file to one machine, one system, or one behavior.
ALWAYS keep snapshot baselines under `snapshots/` and lint baselines under
`snapshots/lint/`.
ALWAYS hard-fail runs when required tools are missing.
ALWAYS enforce line coverage >= 90%.
ALWAYS run `scripts/build_with_zig.sh`, `scripts/test_with_coverage.sh`,
`scripts/lint_snapshot.sh`, and `scripts/bench.sh --snapshot` after each
implementation change.
ALWAYS use ctest targets `emel_tests` and `lint_snapshot` for test execution.
ALWAYS reference `docs/sml.rules.md` for SML semantics and testing guidance.

## Reference Policy
ALWAYS treat `src/` Boost.SML machines as the single source of truth for
architecture and orchestration.
NEVER maintain parallel machine-definition markdown specs under
`docs/architecture/*`.
ALWAYS document state purpose, key invariants, guard semantics, and action side
effects.
ALWAYS treat the reference implementation as the functional logic reference for
allocator and behavioral parity work.
NEVER port reference control flow, branching structure, lifecycle semantics, or
orchestration decisions verbatim from llama.cpp/ggml.
ALWAYS define EMEL behavior and orchestration semantics in Boost.SML machines as
source of truth.
ALWAYS port llama.cpp/ggml arithmetic, kernels, and instruction behavior into
this codebase when implementing equivalent EMEL functionality.
ALWAYS preserve or improve performance when porting logic from the reference
implementation.
ALWAYS implement equivalent functionality natively without external llama.cpp or
ggml linkage.
NEVER link "emel" against llama.cpp or ggml outside `tools/bench`.
ALWAYS link llama.cpp and ggml together with emel in `tools/bench` only.
NEVER use `llama_` or `ggml_` prefixes in identifiers, symbols, files, or APIs.
ALWAYS use `emel_` or `EMEL_` prefixes for project-owned identifiers, symbols,
files, and APIs.
