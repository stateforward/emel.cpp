ALWAYS build a deterministic, production-grade C++ inference engine.
ALWAYS treat performance as a top-level priority in design and implementation decisions.
ALWAYS use Boost.SML state machines as first-class orchestration.
ALWAYS keep public APIs C-compatible and enforce quality gates in CI.
ALWAYS keep public API headers in include/emel/.
ALWAYS keep implementation files in src/.
ALWAYS keep tests in tests/.
ALWAYS map directory layout to namespaces.
ALWAYS colocate machine definition, data, guards, actions, and events within the same component directory.
NEVER place state-machine orchestration logic in data-only files.
ALWAYS use Boost.SML for orchestration state machines.
ALWAYS define trigger intent events in the `event` namespace using noun-like, domain-action names without `cmd_` prefixes (for example `reserve`, `grow`, `recover`, `reset`).
ALWAYS define machine outcome events in the `events` namespace with explicit `_done` and `_error` suffixes.
NEVER use `cmd_*`-prefixed event names.
NEVER add synthetic fault-injection knobs to production events or actions (for example `forced_error`, `requested_status`, `*_error_after`, or per-phase injected error fields).
NEVER add test-only control fields to `src/` machine/event payloads.
ALWAYS model failures from real operation results and route them through explicit `_error` events and error states.
NEVER port machine logic, states, guards, actions, or transitions from `machine/machine.cpp`; treat it as non-functional reference only.
ALWAYS treat `tmp/llama.cpp` as the functional logic reference for allocator/behavioral parity work.
ALWAYS keep guards pure and deterministic.
ALWAYS keep side effects in actions only.
ALWAYS model orchestration decisions with transitions and guards instead of ad-hoc control flow.
ALWAYS process events with run-to-completion semantics.
ALWAYS use boost::sml::back::process for same-machine event chaining from actions, and avoid overriding process_event for internal orchestration.
NEVER call sm.process_event from actions for same-machine dispatch; use boost::sml::back::process instead.
NEVER silently drop unexpected events.
ALWAYS define explicit behavior for unexpected events.
ALWAYS keep machine coupling event-based.
NEVER mutate another machine's context directly.
ALWAYS mutate context inside transition actions, including internal transitions when needed.
ALWAYS define a component-local state-machine context type (for example `action::context`) and inject it into Boost.SML through the machine constructor.
ALWAYS make guards and actions operate on the injected context object, not by mutating `sm` internals directly.
NEVER rely on `friend` access to `sm` private fields for normal guard/action state mutation.
NEVER call another state machine's actions directly.
NEVER call another state machine's guards directly.
NEVER call another state machine's functions directly.
ALWAYS communicate with another state machine only by dispatching events through that machine's `process_event(...)` boundary (directly or via an explicit event sink/mailbox interface).
ALWAYS communicate between state machines through events and explicit interfaces only.
ALWAYS allow passing `process_event` function pointers (or equivalent callable wrappers) through event payloads or context dispatch tables when direct `sm` pointers are unavailable.
ALWAYS prefer boost::sml::back::process for same-machine dispatch and keep sm.process_event at external boundaries or cross-machine calls.
ALWAYS pass explicit machine pointers/references in event payloads for internal machine-to-machine
coordination when ownership/lifetime is guaranteed, and dispatch only via `machine->process_event(...)`.
ALWAYS keep child-machine `data` owned by the parent-machine `data` when composing machines, and inject parent context into children by reference.
ALWAYS have child-to-parent communication dispatch events only via the parent's
`process_event(...)` boundary, never by invoking parent actions, guards, or other functions directly.
ALWAYS define Boost.SML transition tables in `struct model` and expose machine aliases as
`using sm = boost::sml::sm<model>;`.
ALWAYS keep canonical machine types in component namespaces as `emel::<component>::sm`.
ALWAYS provide top-level PascalCase C++ type aliases for orchestration readability
(`emel::Model`, `emel::Parser`, `emel::Generator`, etc.) that alias canonical `emel::<component>::sm` types.
NEVER replace canonical component-scoped `sm` types with top-level aliases; aliases are additive only.
ALWAYS use snake_case for internal SML state type identifiers.
ALWAYS put per-invocation external inputs (other machine pointers, target model pointers, capability
flags, policy flags) on the triggering event payload, not in persistent machine `data`.
ALWAYS keep machine `data` focused on machine-owned runtime state and stable status fields.
NEVER add redundant `status_code` fields to machine context when error is already modeled by explicit error states/events and can be reported through triggering-event outputs (for example `event::...::error_out`).
NEVER store per-invocation API output pointers in machine context (for example `error_out`, `*_out` result pointers, or callback outputs); keep them on triggering event payloads and handle boundary write-back in `sm::process_event(...)`.
NEVER mirror explicit state/event outcomes into context members (for example `is_initialized`, `had_error`, `last_error_recoverable`, phase enums, or duplicated error/result codes) when states and `_done`/`_error` events already represent that information.
ALWAYS use context only for machine-owned runtime data required by actions across internal phase events (for example working buffers, counters, and owned child machines), not for duplicating orchestration status.
NEVER store orchestration phase/attempt/failure flags in machine context (for example `*_state`, `*_attempt`, `*_failure`) when they can be represented by explicit states and events.
ALWAYS encode orchestration retries and one-shot attempts in the transition graph (states + `_done`/`_error` events), not in mutable context flags.
NEVER add string/pointer `error` members to machine `data`; represent failures with explicit error states and `_error` events, and expose boundary status through error codes.
ALWAYS model step outcomes with explicit `_done` / `_error` events and explicit transitions.
NEVER rely on choice pseudostates for normal success/error routing when explicit events are available.
ALWAYS dispatch terminal machine outcome to owner from terminal-state entry actions via
`owner_sm->process_event(...)`.
NEVER call actions directly from API code.
NEVER call guards directly from API code.
NEVER call backend code directly from API code.
ALWAYS route API orchestration through state-machine events only.
ALWAYS inject time, randomness, and external services.
NEVER use globals or singletons in transition logic.
ALWAYS represent errors as explicit machine states or transitions.
ALWAYS classify recoverable and permanent errors through events.
NEVER use exceptions for control flow in hot paths.
ALWAYS use Zig toolchain (zig cc and zig c++) for default development and production builds.
ALWAYS use native clang or gcc for coverage builds.
NEVER rely on compiler-specific behavior without explicit compatibility checks.
ALWAYS use doctest for unit tests.
ALWAYS use SML introspection for machine assertions, including sm.is(...), state visitors, and testing policy.
ALWAYS name test files by machine or domain, such as tests/inference/sampler_tests.cpp.
NEVER use arbitrary or ad-hoc test file names.
NEVER create monolithic test files.
ALWAYS scope each test file to one machine, one system, or one behavior.
ALWAYS keep snapshot baselines under ./snapshots.
ALWAYS fail tests for snapshot regressions unless explicitly updated.
ALWAYS keep lint snapshot baselines under ./snapshots/lint.
ALWAYS hard-fail runs when required tools like clang-format, llvm-cov, llvm-profdata, or gcovr are missing.
ALWAYS enforce line coverage >= 90%.
ALWAYS fail CI and local coverage runs below threshold.
ALWAYS reference docs/sml.md for SML patterns and testing semantics.
ALWAYS ask the user before changing state machine structure.
ALWAYS treat `src/` Boost.SML machines as the single source of truth for architecture and orchestration.
NEVER maintain parallel machine-definition markdown specs under `docs/architecture/*`.
ALWAYS document state purpose, key invariants, guard semantics, and action side effects.
ALWAYS use extern "C" for public API function signatures.
ALWAYS use fixed-width integer types at API boundaries.
ALWAYS return error codes across API boundaries.
NEVER throw exceptions across API boundaries.
NEVER expose C++ templates, classes, or STL containers directly in the public C ABI.
ALWAYS prefer compile-time polymorphism in hot paths.
NEVER use dynamic dispatch in inference hot paths unless explicitly justified.
NEVER allocate in token-generation hot paths.
NEVER use heap allocation by default.
NEVER use heap allocations unless authorized by the user.
ALWAYS prefer stack storage, fixed-capacity containers, or preallocated buffers.
ALWAYS move unavoidable heap allocation outside hot paths.
ALWAYS reuse unavoidable heap allocation.
ALWAYS document rationale for unavoidable heap allocation in code.
ALWAYS require explicit justification and measurable performance rationale for new heap allocations in inference or sampling hot paths.
ALWAYS keep telemetry non-blocking and optional.
ALWAYS use snake_case for functions, variables, and namespaces.
ALWAYS use PascalCase for types and state names.
ALWAYS use SCREAMING_SNAKE_CASE for constants and macros.
ALWAYS keep line length near 100 columns and use 2-space indentation.
NEVER use using namespace in headers.
ALWAYS keep code portable across Linux, macOS, and Windows.
ALWAYS test on x86_64 and arm64 in CI where coverage permits.
NEVER use platform-specific APIs unless wrapped behind an abstraction.
ALWAYS validate numerical behavior and performance against llama.cpp baselines.
ALWAYS maintain GGUF compatibility and versioned state schema migration paths.
ALWAYS build with scripts/build_with_zig.sh for Zig builds.
ALWAYS run coverage and threshold enforcement with scripts/test_with_coverage.sh.
ALWAYS run lint snapshot gate with scripts/lint_snapshot.sh.
ALWAYS run gates after each implementation change: scripts/build_with_zig.sh, scripts/test_with_coverage.sh, and scripts/lint_snapshot.sh.
ALWAYS use ctest targets emel_tests and lint_snapshot for test execution.
ALWAYS continue implementing every required state machine directly in production-quality C++ under `src/` until all are complete.
ALWAYS proceed when implementing state machines and do not ask to continue.
NEVER use scaffolding logic for inference core behavior.
NEVER link llama.cpp.
NEVER link ggml.
ALWAYS treat `tmp/llama.cpp` and ggml sources as numerical/instruction references, not orchestration specifications.
ALWAYS port math, tensor arithmetic, and low-level instruction behavior from references when implementing equivalent EMEL functionality.
NEVER port reference control flow, branching structure, lifecycle semantics, or orchestration decisions verbatim from llama.cpp/ggml.
ALWAYS define EMEL behavior and orchestration semantics in Boost.SML state machines (states, events, guards, actions) as the source of truth.
ALWAYS prioritize EMEL state-machine semantics over branch-for-branch parity with reference implementations.
ALWAYS target numerical and functional outcome parity where applicable, without mirroring reference branching patterns.
ALWAYS port llama.cpp and ggml arithmetic into this codebase.
ALWAYS port llama.cpp and ggml instruction behavior into this codebase.
ALWAYS port arithmetic, kernels, and inference instructions directly from tmp/llama.cpp into this codebase.
ALWAYS preserve or improve performance when porting logic from tmp/llama.cpp.
ALWAYS implement equivalent functionality natively in this project without external llama.cpp or ggml linkage.
NEVER use llama_ prefix in identifiers, symbols, files, or APIs.
NEVER use ggml_ prefix in identifiers, symbols, files, or APIs.
ALWAYS use emel_ or EMEL_ prefixes for project-owned identifiers, symbols, files, and APIs.
