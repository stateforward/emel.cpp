# Phase 12: Parity And Verification Closure - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 12 proves that the shipped flash-attention path is truthfully exercised on the maintained
`tools/paritychecker --generation` surface for the canonical Llama-68M slice and remains parity
stable against the latest synced `llama.cpp` reference. This phase stays inside parity and test
closure: it does not broaden public/runtime surface area, and it does not publish benchmark
evidence yet.

</domain>

<decisions>
## Implementation Decisions

### Reference Alignment Target
- **D-01:** Phase 12 parity must build the reference implementation through CMake against the
  latest upstream `llama.cpp` state instead of reading from a local `tmp/llama.cpp` checkout.
- **D-02:** If the latest upstream `llama.cpp` changes behavior and parity breaks, that drift blocks
  Phase 12 completion instead of being treated as non-blocking information.
- **D-03:** Local `tmp/llama.cpp` state must not decide parity results for this phase; the parity
  surface should be truthful even on machines that do not have that local checkout.
- **D-04:** Parity success still means exact token/output parity on the maintained bounded
  workloads, not loose behavioral similarity.

### Flash Proof Surface
- **D-05:** The normal `paritychecker --generation` success surface must expose flash-execution
  proof directly; proof must not live only behind an explicit dump mode.
- **D-06:** Phase 12 may introduce a clearer dedicated proof block on normal output rather than
  strictly preserving the prior success-line shape.
- **D-07:** The maintained proof surface must expose at least flash-dispatch call counts so the
  proof is explicit and machine-checkable.
- **D-08:** When parity fails or flash proof is missing, the tool should auto-emit enough
  diagnostics to explain the failure without requiring a second rerun just to inspect the seam.

### Verification Contract
- **D-09:** Phase 12 should gate on both the existing canonical short generation request and one
  bounded longer decode on the same maintained Llama-68M workload contract.
- **D-10:** Both bounded workloads must pass; if the short run passes but the longer bounded decode
  diverges, Phase 12 is still blocked.
- **D-11:** The phase should remain success-path focused on the paritychecker surface; broader
  unsupported/failure-matrix coverage continues to live primarily in existing kernel and generator
  tests unless one narrow paritychecker-level failure contract falls out naturally.

### the agent's Discretion
- Exact wording, formatting, and field names for the normal proof block and failure diagnostics.
- Exact bounded-long decode shape, as long as it remains the maintained canonical Llama-68M slice
  and stays intentionally bounded.
- Exact mechanics for refreshing `tools/paritychecker/reference_ref.txt` and any adjacent test
  assertions once latest-reference validation passes.

</decisions>

<specifics>
## Specific Ideas

- The parity build should make its upstream reference choice visible enough that users can tell it
  came from the CMake-fetched latest `llama.cpp` path rather than a repo-local checkout.
- It is acceptable for Phase 12 to make the normal paritychecker output more explicit and slightly
  more verbose if that produces a clearer flash-proof surface.
- Keep the milestone honest by proving flash execution and parity on the existing paritychecker
  surface rather than inventing new CLI knobs or a separate proof-only tool path.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone Scope And Rules
- `.planning/ROADMAP.md` — Phase 12 goal, success criteria, and dependency on Phase 11.
- `.planning/REQUIREMENTS.md` — `PAR-01`, `PAR-02`, and `VER-01`, plus the explicit out-of-scope
  limits for this milestone.
- `AGENTS.md` — project engineering contract, including performance, SML, and flash-truthfulness
  constraints.
- `docs/rules/sml.rules.md` — RTC/no-queue semantics that still govern any orchestration-adjacent
  changes.

### Prior Phase Outputs
- `.planning/phases/10-flash-kernel-bring-up/10-CONTEXT.md` — canonical-only flash scope and
  explicit rejection policy established in Phase 10.
- `.planning/phases/10-flash-kernel-bring-up/10-VERIFICATION.md` — kernel-local proof already
  shipped and not to be re-scoped here.
- `.planning/phases/11-generator-flash-adoption/11-CONTEXT.md` — generator flash-adoption
  decisions, especially flash-dispatch observability and truthful unsupported behavior.
- `.planning/phases/11-generator-flash-adoption/11-VERIFICATION.md` — proof that generator-owned
  flash dispatch counters already exist and are trustworthy inputs for Phase 12.

### Parity Surface And Reference Alignment
- `tools/paritychecker/parity_runner.cpp` — current generation parity contract, flash-proof seam,
  and failure-reporting surface.
- `tools/paritychecker/paritychecker_tests.cpp` — subprocess-facing assertions for normal output,
  diagnostics, and generation contract behavior.
- `tools/paritychecker/CMakeLists.txt` — reference source-selection policy for paritychecker and
  where the "fetch latest upstream, do not use local tmp checkout" contract must live.

### Runtime And Fixture Inputs
- `src/emel/generator/detail.hpp` — generator flash request formation and current runtime attention
  seam used by paritychecker.
- `src/emel/generator/sm.hpp` — exported generator observability getters, including flash-dispatch
  proof counters.
- `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` — fixed model fixture for the maintained parity
  contract.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/paritychecker/parity_runner.cpp::run_generation_harness_contract(...)` already owns the
  canonical success/failure contract for `--generation` and is the narrowest place to harden the
  Phase 12 proof rules.
- `tools/paritychecker/parity_runner.cpp::dump_reference_decode_seam(...)` already exists as the
  diagnostic seam for failure analysis and can be made the automatic failure explainer without
  inventing a new surface.
- `src/emel/generator/sm.hpp` already exposes
  `generation_flash_attention_dispatch_calls()`, which is the durable runtime proof surface added
  in Phase 11.
- `tools/paritychecker/paritychecker_tests.cpp` already captures stdout/stderr and parses named
  metrics, so normal-output proof formatting can be enforced in subprocess tests instead of only in
  local helper assertions.

### Established Patterns
- Prior phases already fixed that flash claims must stay truthful: no silent fallback and no
  relabeling the old path as flash.
- The milestone boundary remains narrow: parity proof lives on `tools/paritychecker`, benchmark
  evidence remains deferred to Phase 13, and no new public/API surface should appear here.
- Kernel-local correctness and generator-local unsupported behavior already have dedicated tests, so
  Phase 12 does not need to explode into a broad paritychecker failure matrix unless one narrow
  contract test is clearly justified.

### Integration Points
- `tools/paritychecker/parity_runner.cpp` is the integration seam for combining latest-reference
  validation, flash-proof publication, and bounded generation parity checks.
- `tools/paritychecker/paritychecker_tests.cpp` is the integration seam for locking the exact
  subprocess-visible proof and diagnostic contract.
- `tools/paritychecker/CMakeLists.txt` is the integration seam for making parity builds fetch the
  current upstream reference instead of consuming repo-local `tmp` state.

</code_context>

<deferred>
## Deferred Ideas

- Benchmark compare rows, artifact refresh, and measurable flash-performance publication belong to
  Phase 13.
- Broader multi-model or non-canonical flash parity remains out of scope for this phase.
- A broad paritychecker failure matrix for many unsupported cases is deferred unless later work
  proves the single success-path contract is insufficient.

</deferred>

---
*Phase: 12-parity-and-verification-closure*
*Context gathered: 2026-03-21*
