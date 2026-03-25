# Codebase Concerns

## Current Risk Profile

The repository is explicitly unstable right now. `README.md` marks the project as WIP, and
`docs/plans/rearchitecture.plan.md` still describes a draft hard-cutover migration with legacy
cleanup, scaffold replacement, and gate restoration still in flight. That makes regressions more
likely in integration-heavy components such as `src/emel/generator/*`, `src/emel/graph/*`,
`src/emel/text/*`, and the build/test manifests in `CMakeLists.txt` and `scripts/*`.

The repo is also large relative to its maturity level: `src/emel` contains hundreds of source
files, there are 80+ `sm.hpp` machine files under `src/emel/**/sm.hpp`, `docs/architecture/*`
contains a large generated surface, and `tests/models` alone is about 614 MB. That combination
raises the cost of keeping architecture, tests, docs, and benchmarks in sync.

## Highest-Risk Concerns

- `scripts/quality_gates.sh` is not a full gate right now. It does not run
  `scripts/test_with_sanitizers.sh`, it has `scripts/lint_snapshot.sh` commented out, it treats
  benchmark regressions as warnings, and it runs only smoke-level fuzzing via `scripts/fuzz_smoke.sh`.
  Memory-safety bugs, lint drift, and performance regressions can therefore slip through.

- `scripts/test_with_coverage.sh` excludes `src/emel/**/sm.hpp` from coverage accounting and only
  enforces `BRANCH_COVERAGE_MIN=50`. The most important orchestration logic lives in
  `src/emel/**/sm.hpp`, so the headline coverage number overstates confidence in the actor layer.

- `docs/compliance-report.md` already records unresolved structural debt: a source-first transition
  offender in `src/emel/text/formatter/sm.hpp`, many machines without public wrappers, and remaining
  runtime-conditional debt. The report is useful, but it is a generated static snapshot rather than
  an always-on gate, so it can drift from HEAD.

- `.github/workflows/gemini-dispatch.yml`, `.github/workflows/gemini-review.yml`, and
  `.github/workflows/gemini-invoke.yml` automate AI review/triage flows, but there is no visible
  first-class GitHub Actions pipeline for build, coverage, sanitizer, fuzz, parity, and docs
  enforcement. The repository currently leans on local discipline more than hosted CI.

## Fragile Areas And Likely Bug Vectors

- `src/emel/sm.hpp` is a high-leverage risk concentration. It implements event-result
  normalization, custom type erasure in `sm_any`, placement-new lifetime management, `std::launder`
  recovery, enum-to-index dispatch, and visitor routing. A defect here propagates into
  `src/emel/kernel/any.hpp`, `src/emel/text/encoders/any.hpp`, and
  `src/emel/text/tokenizer/preprocessor/any.hpp`.

- `src/emel/sm.hpp` silently normalizes invalid kind enums to the default index in `sm_any`. That
  avoids crashes, but it can also hide configuration bugs by routing bad values to the first
  backend in `src/emel/kernel/any.hpp` or the first encoder in `src/emel/text/encoders/any.hpp`.

- `src/emel/sm.hpp` still carries `process_support::immediate_queue`, and
  `tests/sm/sm_policy_tests.cpp` exercises it. Even if it exists only as compatibility glue, it
  sits uncomfortably close to the no-queue invariant described in `docs/rules/sml.rules.md`.

- `src/emel/generator/sm.hpp`, `src/emel/generator/actions.hpp`, and `src/emel/generator/guards.hpp`
  are the densest integration point in the repository. They coordinate `text`, `memory`, `graph`,
  and `logits` machines, so small contract drift in one subsystem can break end-to-end generation.

- `src/emel/generator/guards.hpp` centralizes cross-component error classification for
  `src/emel/text/conditioner/errors.hpp`, `src/emel/text/renderer/errors.hpp`,
  `src/emel/memory/hybrid/errors.hpp`, `src/emel/graph/errors.hpp`, and
  `src/emel/logits/sampler/errors.hpp`. That is convenient, but it is also fragile: enum changes
  or new error codes can be misclassified without a compiler failure.

- `src/emel/text/renderer/actions.hpp` uses arithmetic selection, table-style routing, and
  hand-rolled bounds logic to stay within the branch-model rules. That preserves the SML contract,
  but it makes stop-sequence matching, holdback accounting, and output composition harder to audit
  for off-by-one and buffer-edge bugs.

- `src/emel/text/formatter/sm.hpp` is still a scaffold with embedded draft design text and the
  compliance-reported source-first transition syntax. Anything depending on it should be treated as
  unstable until the formatter boundary is finalized.

- Callback-based same-RTC handoff is used in multiple places, including `src/emel/gguf/loader/actions.hpp`,
  `src/emel/generator/actions.hpp`, `src/emel/logits/sampler/events.hpp`, and
  `tests/sm/callback_tests.cpp`. This is valid under the project rules, but lifetime mistakes here
  will be subtle because they sit between stack-only event objects and nested synchronous dispatch.

## Performance And Memory Concerns

- `src/emel/text/unicode.hpp` is a major performance concern. It uses `std::regex`, `std::string`,
  `std::vector`, `std::map`, `std::unordered_map`, and throws exceptions. This sits on the text
  and tokenizer path, so the current implementation is far from the repo's long-term "no allocation
  in hot paths" target.

- `src/emel/text/encoders/plamo2/context.hpp` and `src/emel/text/encoders/plamo2/detail.hpp` use
  `std::unordered_map`, `std::vector`, and `std::string` extensively. That may be acceptable during
  setup, but it is a risk until proven to be outside latency-critical encode dispatch.

- `src/emel/gbnf/rule_parser/detail.hpp` keeps dynamic containers for parser state, and
  `src/emel/gguf/loader/*` plus `src/emel/text/jinja/*` add more parsing-heavy code paths. The
  project can still be correct with these choices, but deterministic throughput and allocation
  behavior are not yet convincingly locked down.

- `scripts/quality_gates.sh` currently allows up to 30% benchmark variance and suppresses a failing
  benchmark step into a warning. Combined with the broad benchmark surface in `tools/bench/*`, that
  means performance drift can accumulate before the team gets a hard failure.

- `tests/models` and `tmp/test_models` add substantial binary weight to the working tree. The model
  list in `tests/models/README.md` is useful, but the current asset strategy increases clone size,
  local disk pressure, and CI cache churn.

## Security And Supply-Chain Concerns

- `CMakeLists.txt` fetches Boost.SML from the network via `cmake/sml_version.cmake`, and both
  `tools/bench/CMakeLists.txt` and `tools/paritychecker/CMakeLists.txt` fetch `llama.cpp` unless
  `tmp/llama.cpp` is present. The refs are pinned, which helps, but builds are still dependent on
  external repository availability and integrity.

- The untrusted-input surface is broad: `src/emel/gguf/loader/*` parses model files,
  `src/emel/gbnf/*` parses grammars, `src/emel/text/jinja/*` parses templates, and
  `src/emel/text/unicode.hpp` processes user/model text. Those are natural bug and security
  boundaries for malformed input.

- Fuzzing exists, but `tests/fuzz/gguf_parser_fuzz.cpp`, `tests/fuzz/gbnf_parser_fuzz.cpp`,
  `tests/fuzz/jinja_parser_fuzz.cpp`, and `tests/fuzz/jinja_formatter_fuzz.cpp` only cover part of
  the attack surface, and `scripts/fuzz_smoke.sh` gives each target a short smoke run. There is no
  comparable fuzz target for `src/emel/text/unicode.hpp`, `src/emel/text/renderer/*`, or the
  generator-to-text glue paths.

## Missing Automation And Process Debt

- `docs/runtime-conditionals-todo.md` says the listed action/detail files are complete, while
  `docs/compliance-report.md` still reports runtime-conditional offenders. That mismatch suggests
  compliance tracking is not yet a single trustworthy source of truth.

- `scripts/generate_docs.sh` and `tools/docsgen/docsgen.cpp` create a large generated-doc surface in
  `docs/architecture/*`. The generation story exists, but without a strong CI gate the repo can
  carry stale architecture docs for long stretches.

- `README.md` tells contributors to use `scripts/quality_gates.sh`, but the same script contains
  temporary disable comments tied to the rearchitecture. The policy says "run the gates", while the
  implementation says "some gates are knowingly not restored yet".

- `docs/plans/rearchitecture.plan.md` explicitly says the cutover is still deleting legacy tests,
  benches, and adapters. Until that plan is closed, `CMakeLists.txt`, `tests/*`, `tools/bench/*`,
  and `docs/architecture/*` should all be treated as potentially carrying transitional residue.

## Architectural Risks

- The project is intentionally "state machines everywhere", which makes behavior explicit but also
  creates a large number of tiny contracts. When the architecture is still moving, small changes in
  `src/emel/**/events.hpp`, `src/emel/**/errors.hpp`, and `src/emel/**/context.hpp` can cascade
  widely across the tree.

- There is a tension between the actor-model purity in `docs/rules/sml.rules.md` and the practical
  need for fast text, parser, and template code in `src/emel/text/*` and `src/emel/gbnf/*`. The
  repo has not fully resolved that tension yet; parts of the text stack still look like classic STL
  utilities wrapped by SML orchestration rather than strictly allocation-free actor kernels.

- The repo treats `src/emel/**/sm.hpp` docstrings as architecture contracts, but also generates
  `docs/architecture/*` and keeps design-plan material in `docs/plans/rearchitecture.plan.md`.
  That creates a three-way drift risk between source comments, generated docs, and planning docs.

- The hardest part of the system is not any single actor; it is the composition chain running from
  `src/emel/gguf/loader/*` through `src/emel/text/*`, `src/emel/memory/*`, `src/emel/graph/*`, and
  into `src/emel/generator/*`. That path is where subtle semantic mismatches are most likely to
  appear first.

## Suggested Stabilization Priorities

- Restore hard automation first in `scripts/quality_gates.sh`, especially `scripts/test_with_sanitizers.sh`,
  `scripts/lint_snapshot.sh`, and hard benchmark failure behavior.

- Treat `src/emel/sm.hpp` as critical infrastructure and keep additional complexity out of it until
  the `sm_any` and callback surfaces are better proven.

- Put stricter measurement around the text stack, especially `src/emel/text/unicode.hpp`,
  `src/emel/text/encoders/*`, and `src/emel/text/renderer/*`, because those files are the clearest
  gap between the performance contract and current implementation style.

- Close the documentation/compliance loop so `docs/compliance-report.md`, `docs/runtime-conditionals-todo.md`,
  `docs/architecture/*`, and `docs/plans/rearchitecture.plan.md` stop disagreeing about repo state.
