# Phase 165: Benchmark Actor Boundary Enforcement Closure - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

Close `LANE-02` by making benchmark runner sources stop reaching directly into actor
`actions.hpp`, `guards.hpp`, or actor-owned `detail.hpp` helper surfaces, then broaden source
checks so maintained runner sources are scanned, not just shared orchestration files.

</domain>

<decisions>
## Implementation Decisions

### Actor Boundary Scope
- Treat direct `::action::` / `::guard::` namespace use in `tools/bench` runner sources as
  prohibited.
- Treat actor component detail surfaces with owned `sm.hpp` orchestration as prohibited for
  maintained runner sources when the bench can use public events, public state-machine wrappers,
  or non-actor kernel/model diagnostic surfaces instead.
- Keep non-actor numeric/kernel/detail usage allowed where it is the maintained benchmark's
  execution surface, such as kernel quant helpers, tokenizer internals, and model/fixture loader
  diagnostic helpers.

### Source Changes
- Remove direct action-context construction from jinja parser/formatter benches by using public
  state-machine wrappers.
- Remove direct batch planner action-constant reach-through by deriving local capacities from the
  public event scratch type.
- Remove diarization pipeline/executor actor detail use from maintained bench sources where public
  errors/events and non-actor output detail constants provide the same checks.

### Verification Strategy
- Add a broad source-backed test that recursively scans maintained `tools/bench` `.cpp`/`.hpp`
  runner sources for actor internal include and namespace patterns.
- Keep the existing shared orchestration lane-neutral check.
- Re-run focused source tests, generation/diarization behavior tests, full unfiltered
  `bench_runner_tests`, domain-boundary checks, and a changed-file scoped quality gate.

</decisions>

<code_context>
## Existing Code Insights

### Missed Reach-Through
- `tools/bench/text/jinja/parser_bench.cpp` and `formatter_bench.cpp` constructed
  `parser::action::context` and `formatter::action::context` directly.
- `tools/bench/batch/planner_bench.cpp` read `emel::batch::planner::action::MAX_PLAN_STEPS`.
- `tools/bench/diarization/sortformer_bench.cpp` and
  `tools/bench/diarization/sortformer_fixture.hpp` used diarization request/pipeline/executor
  detail surfaces that sit beside actor `sm.hpp` files.

### Existing Guardrail Gap
- `bench_runner_tests.cpp` only scanned shared benchmark files for actor internal patterns.
- Maintained runner implementation files under `tools/bench/**` were outside that scan.

</code_context>

<deferred>
## Deferred Ideas

Longer-term public diagnostic APIs for model/loader fixture construction can be designed in a
future milestone. This phase only closes the actor-boundary gap identified by the v1.19 audit.

</deferred>
