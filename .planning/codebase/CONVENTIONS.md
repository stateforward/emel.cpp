# EMEL Conventions

This repository is governed first by `docs/rules/sml.rules.md`, then by `AGENTS.md`, with
`docs/rules/cpp.rules.md` supplying broader C/C++ real-time constraints. The implementation
style in `src/` is heavily shaped by Boost.SML orchestration, synchronous actor boundaries,
and “no allocation during dispatch” expectations.

## Coding Style

- Language level is `C11` and `C++20` in `CMakeLists.txt`.
- Warnings are treated as errors in `CMakeLists.txt` via `-Wall -Wextra -Wpedantic -Werror`
  on non-MSVC toolchains and `/W4 /WX` on MSVC.
- Internal code uses `snake_case` for namespaces, functions, variables, states, and events per
  `AGENTS.md` and visible examples such as `src/emel/gbnf/sampler/sm.hpp`,
  `src/emel/graph/allocator/events.hpp`, and `src/emel/batch/planner/sm.hpp`.
- Exported/public C++ type aliases use PascalCase, usually as additive aliases over `sm`, for
  example `using Sampler = sm;` in `src/emel/gbnf/sampler/sm.hpp`,
  `using Allocator = sm;` in `src/emel/graph/allocator/sm.hpp`, and
  `using Planner = sm;` in `src/emel/batch/planner/sm.hpp`.
- Public C ABI types use C naming and fixed-width integers, as shown in `include/emel/emel.h`.
- Headers use `#pragma once` throughout `src/emel/**` and conventional include guards at the C
  boundary in `include/emel/emel.h`.
- Formatting is intentionally stabilized around transition tables with narrow `// clang-format off`
  blocks in files such as `src/emel/gbnf/sampler/sm.hpp`, `src/emel/graph/allocator/sm.hpp`, and
  `src/emel/batch/planner/sm.hpp`.
- Line length and indentation expectations come from `AGENTS.md`: near 100 columns and 2-space
  indentation.
- `using namespace` in headers is prohibited by `AGENTS.md`; the code instead uses local aliases
  like `namespace sml = boost::sml;` inside functions in `src/emel/*/sm.hpp`.

## Naming Rules

- Repository-owned prefixes are `emel_` or `EMEL_`; `AGENTS.md` explicitly forbids `llama_` and
  `ggml_` prefixes outside `tools/bench` and `tools/paritychecker`.
- Trigger events live in `event` namespaces and outcome/internal events live in `events`
  namespaces, for example `src/emel/graph/allocator/events.hpp`,
  `src/emel/gbnf/sampler/events.hpp`, and `src/emel/batch/planner/events.hpp`.
- Outcome event names are expected to end in `_done` or `_error`; current examples include
  `sample_done` and `sample_error` in `src/emel/gbnf/sampler/events.hpp` and
  `allocation_done` and `allocation_error` in `src/emel/graph/allocator/events.hpp`.
- State labels are lower snake case structs such as `ready`, `liveness_decision`,
  `mode_decision`, `publishing`, and `errored` in `src/emel/graph/allocator/sm.hpp`,
  `src/emel/batch/planner/sm.hpp`, and `src/emel/gbnf/sampler/sm.hpp`.
- Test file names are domain or machine specific, for example
  `tests/text/tokenizer/tokenizer_tests.cpp`, `tests/graph/allocator/allocator_tests.cpp`, and
  `tests/batch/planner/planner_sm_transition_tests.cpp`; this matches the naming rule in
  `AGENTS.md`.

## Architecture And Composition

- `src/` is the source of truth for architecture per `AGENTS.md`; there is no parallel
  architecture spec tree to consult first.
- The dominant pattern is one component directory per machine, with colocated
  `actions.hpp`, `context.hpp`, `events.hpp`, `guards.hpp`, `errors.hpp`, `sm.hpp`, and optional
  `detail.hpp`, as seen in `src/emel/gbnf/sampler/`, `src/emel/graph/allocator/`, and
  `src/emel/text/tokenizer/preprocessor/bpe/`.
- `src/emel/gbnf` is the stated default architectural reference family in `AGENTS.md`, and it is
  a good exemplar for machine decomposition and transition-table layout.
- Every machine defines a `struct model` transition table and usually a wrapper `struct sm :
  public emel::sm<model, action::context>` in `src/emel/*/sm.hpp`.
- Machine wrappers own or inject context and provide a narrow public `process_event` surface; see
  `src/emel/graph/allocator/sm.hpp`, `src/emel/batch/planner/sm.hpp`, and
  `src/emel/gbnf/sampler/sm.hpp`.
- Cross-machine communication is synchronous and explicit through `process_event(...)`, for
  example in `src/emel/kernel/actions.hpp`, `src/emel/graph/actions.hpp`,
  `src/emel/graph/assembler/assemble_alloc_pass/actions.hpp`, and `src/emel/memory/hybrid/actions.hpp`.
- Shared behavior is expected to live in helpers such as `detail.hpp`, `actions.hpp`, or `sm_any`
  infrastructure in `src/emel/sm.hpp`, not in inheritance hierarchies.
- Public wrapper visibility is part of the design contract and is tested in
  `tests/graph/wrapper_visibility_tests.cpp` and `tests/kernel/lifecycle_tests.cpp`.

## State-Machine Rules

- The top-level semantics are synchronous run-to-completion actor semantics from
  `docs/rules/sml.rules.md` and `AGENTS.md`.
- Queueing is forbidden by policy: `sml::process_queue`, `sml::defer_queue`, and mailbox-style
  “post later” patterns are not allowed by `docs/rules/sml.rules.md` and `AGENTS.md`.
- Transition tables must use destination-first rows:
  `sml::state<dst> <= src + event [guard] / action`. Representative files are
  `src/emel/gbnf/sampler/sm.hpp`, `src/emel/graph/allocator/sm.hpp`, and
  `src/emel/batch/planner/sm.hpp`.
- Transition tables are visually sectioned with divider comments and leading-comma style after the
  first row, again visible in `src/emel/gbnf/sampler/sm.hpp` and `src/emel/graph/allocator/sm.hpp`.
- Runtime control flow belongs in guards and explicit states, not inside actions. This rule is
  stated in `docs/rules/sml.rules.md` and `AGENTS.md` and is reinforced by dedicated branch-path
  tests like `tests/graph/allocator/allocator_action_branch_tests.cpp`,
  `tests/graph/assembler/assembler_action_branch_tests.cpp`,
  `tests/batch/planner/planner_action_branch_tests.cpp`, and
  `tests/graph/processor/processor_action_branch_tests.cpp`.
- Internal multi-step flows should use typed runtime/internal events and `sml::completion<TEvent>`
  rather than self-dispatch. This is the dominant pattern in `src/emel/gbnf/sampler/sm.hpp`,
  `src/emel/graph/allocator/sm.hpp`, `src/emel/batch/planner/sm.hpp`, and
  `src/emel/text/tokenizer/preprocessor/*/sm.hpp`.
- Unexpected external events must be handled explicitly with `sml::unexpected_event`, as seen in
  nearly every machine table, including `src/emel/gbnf/sampler/sm.hpp`,
  `src/emel/graph/allocator/sm.hpp`, and `src/emel/batch/planner/sm.hpp`.
- External state inspection should use `is(...)` or `visit_current_states`, which is exposed by
  wrapper types in files such as `src/emel/gbnf/sampler/sm.hpp` and exercised in
  `tests/text/jinja/parser_tests.cpp`, `tests/text/jinja/formatter_tests.cpp`,
  `tests/batch/planner/planner_sm_flow_tests.cpp`, and `tests/generator/lifecycle_tests.cpp`.

## Event, Context, And Data Flow Norms

- Publicly exposed events should stay small, immutable, and fixed-width where possible, while
  internal same-RTC handoff events may use references or pointers; compare `include/emel/emel.h`
  with runtime/internal structs in `src/emel/gbnf/sampler/events.hpp` and
  `src/emel/graph/allocator/events.hpp`.
- Required event fields should be references according to `AGENTS.md`; `src/emel/gbnf/sampler/events.hpp`
  follows this for `sample`, while some older public request types such as
  `src/emel/graph/allocator/events.hpp` still use pointer outputs and callbacks because they cross
  a dispatch boundary.
- Context should hold persistent actor-owned state, not per-dispatch request mirrors, per
  `docs/rules/sml.rules.md` and `AGENTS.md`.
- Current code shows two patterns:
  the preferred persistent context form in files like `src/emel/graph/allocator/context.hpp`, and
  the older internal per-dispatch runtime-context form in files like
  `src/emel/graph/allocator/events.hpp` and `src/emel/gbnf/sampler/events.hpp`.
- That split is an active quality hotspot: current wrappers often create `*_ctx` or runtime event
  structs inside `process_event(...)` in `src/emel/graph/allocator/sm.hpp`,
  `src/emel/batch/planner/sm.hpp`, `src/emel/text/tokenizer/preprocessor/bpe/sm.hpp`, and
  `src/emel/gbnf/sampler/sm.hpp`, while the policy docs push new work toward typed phase events and
  away from stuffing dispatch-local control state into context.

## Error Handling Patterns

- Errors are modeled as explicit enums and explicit `_error` events or error states rather than
  exceptions. See `src/emel/gbnf/sampler/errors.hpp`, `src/emel/graph/allocator/errors.hpp`, and
  `src/emel/batch/planner/errors.hpp`.
- Public API boundaries should return error codes and use `extern "C"` with fixed-width integers,
  as shown in `include/emel/emel.h` and mandated by `AGENTS.md`.
- Wrapper `process_event(...)` functions generally normalize a boolean acceptance result together
  with an error code in a runtime context, for example in `src/emel/gbnf/sampler/sm.hpp`,
  `src/emel/graph/allocator/sm.hpp`, and `src/emel/batch/planner/sm.hpp`.
- `src/emel/sm.hpp` centralizes generic normalization behavior with
  `emel::detail::normalize_event_result(...)`, which is covered by `tests/sm/sm_policy_tests.cpp`.
- Callback-based same-RTC replies are a repository pattern, especially in graph and planner code,
  for example `src/emel/graph/allocator/events.hpp`, `src/emel/batch/planner/events.hpp`,
  `tests/graph/allocator/allocator_tests.cpp`, and `tests/batch/planner/planner_action_branch_tests.cpp`.
- Unexpected-event handlers commonly mark internal error state or write error outputs, as seen in
  `src/emel/gbnf/sampler/actions.hpp` and many sibling `actions.hpp` files.
- Exceptions are effectively off-limits for control flow in dispatch-critical code per
  `docs/rules/cpp.rules.md`, `docs/rules/sml.rules.md`, and `AGENTS.md`.

## Performance And Portability Norms

- Performance is a first-class design goal in `AGENTS.md`: no heap allocation in inference or
  sampling hot paths, limited dynamic dispatch, and one-time allocation only during setup.
- Default build and production toolchain is Zig via `scripts/build_with_zig.sh`; coverage builds
  intentionally switch to `gcc` and `g++` in `scripts/test_with_coverage.sh`; sanitizer builds use
  `clang` and `clang++` in `scripts/test_with_sanitizers.sh`.
- Portability across Linux, macOS, and Windows is an explicit rule in `AGENTS.md`; scripts
  generally probe `nproc`, `getconf`, and `sysctl` to stay Unix-portable, as in
  `scripts/test_with_coverage.sh` and `scripts/test_with_sanitizers.sh`.
- External reference implementations are quarantined to `tools/bench` and `tools/paritychecker`
  per `AGENTS.md`; that boundary is reflected in `tools/bench/CMakeLists.txt`,
  `tools/paritychecker/CMakeLists.txt`, `scripts/bench.sh`, and `scripts/paritychecker.sh`.

## Repository-Specific Engineering Norms

- The expected post-change validation command is `scripts/quality_gates.sh`, as stated in
  `AGENTS.md` and implemented in `scripts/quality_gates.sh`.
- Snapshot baselines are repository artifacts under `snapshots/`, specifically
  `snapshots/bench/benchmarks.txt`, `snapshots/bench/benchmarks_compare.txt`,
  `snapshots/lint/clang_format.txt`, and `snapshots/quality_gates/timing.txt`.
- Benchmarks are machine-scoped sources in `tools/bench`, for example
  `tools/bench/memory/kv_bench.cpp`, `tools/bench/logits/sampler_bench.cpp`, and
  `tools/bench/gbnf/rule_parser_bench.cpp`.
- Fuzz targets live under `tests/fuzz` and are opt-in through `EMEL_ENABLE_FUZZ` in
  `CMakeLists.txt`; smoke execution is scripted by `scripts/fuzz_smoke.sh`.
- Generated documentation is treated as a quality gate through `scripts/generate_docs.sh` and the
  `generate_docs` ctest registered in `CMakeLists.txt`.

## Quality Hotspots To Keep In Mind

- `AGENTS.md` says to ask before changing state-machine structure, so even apparently mechanical
  state-graph rewrites should be treated as design changes.
- `scripts/quality_gates.sh` currently skips `lint_snapshot` and sanitizer execution, so policy is
  stricter than current gate enforcement.
- `CMakeLists.txt` comments out legacy processor tests from the main test binary, and two current
  test files, `tests/graph/processor/processor_sm_transition_tests.cpp` and
  `tests/graph/processor/processor_tests.cpp`, are present in `tests/` but not registered in
  `EMEL_TEST_SOURCES`.
- Some code still uses dispatch-local runtime context structs in event headers, such as
  `src/emel/gbnf/sampler/events.hpp` and `src/emel/graph/allocator/events.hpp`, which is worth
  reviewing against the stricter context rules in `AGENTS.md` and `docs/rules/sml.rules.md`.
