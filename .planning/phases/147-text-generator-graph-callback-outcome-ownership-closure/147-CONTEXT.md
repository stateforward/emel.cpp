# Phase 147: Text Generator Graph Callback Outcome Ownership Closure - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning
**Mode:** Autonomous gap closure

<domain>
## Phase Boundary

Close the remaining v1.17 `TEXTGEN-04` and `TEXTGEN-07` audit blocker. The maintained
text generator path must not decide graph validation, bind, extract, callback, or error-channel
outcomes through action-called `text/generator/detail.hpp` callback return values or `err_out`
writes.

</domain>

<decisions>
## Implementation Decisions

### Locked Decisions
- Treat the milestone audit as source-backed truth: artifacts alone cannot satisfy this phase.
- Keep runtime behavior choice in Boost.SML guards and destination-first transition rows.
- Keep graph compute readiness checks in parent/prefill guards before graph dispatch.
- Keep `detail.hpp` callbacks limited to already-accepted binding/copying for the maintained path.
- Do not substitute benchmark, model, snapshot, fixture, or tool-only evidence for source repair.

### The Agent's Discretion
- Use the smallest source change that removes the callback outcome bypass.
- Update direct detail tests where they still encode the rejected callback-validation pattern.
- Update model, snapshot, benchmark, or fixture evidence only if maintained-path validation requires it.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Rule Set
- `AGENTS.md` - Project-level SML, detail-helper, context, and quality-gate contract.
- `docs/rules/sml.rules.md` - Normative Boost.SML RTC and routing rules.

### Audit Evidence
- `.planning/v1.17-MILESTONE-AUDIT.md` - Current source-backed gap report.
- `.planning/ROADMAP.md` - Phase 147 gap-closure scope and success criteria.
- `.planning/REQUIREMENTS.md` - `TEXTGEN-04` and `TEXTGEN-07` reopened requirement state.

### Source Under Audit
- `src/emel/text/generator/actions.hpp` - Maintained generator graph dispatch wiring.
- `src/emel/text/generator/detail.hpp` - Generator callback implementations under audit.
- `src/emel/text/generator/guards.hpp` - Parent generator compute readiness predicates.
- `src/emel/text/generator/prefill/guards.hpp` - Prefill compute readiness predicates.
- `src/emel/text/generator/sm.hpp` - Parent generator transition ownership proof.
- `src/emel/text/generator/prefill/sm.hpp` - Prefill transition ownership proof.
- `src/emel/graph/processor/*_step/actions.hpp` - Generic graph callback outcome capture.

### Tests and Evidence
- `tests/text/generator/lifecycle_tests.cpp` - Source-regression scans and maintained actor proof.
- `tests/text/generator/action_guard_tests.cpp` - Guard readiness classification tests.
- `tests/text/generator/detail_tests.cpp` - Detail callback and numeric helper regressions.
- `tools/paritychecker/parity_runner.cpp` - Public parity entrypoint evidence.
- `tools/bench/generation_bench.cpp` - Public benchmark entrypoint evidence.
- `tools/embedded_size/emel_probe/main.cpp` - Embedded-size public entrypoint evidence.

</canonical_refs>

<specifics>
## Specific Ideas

- Refactor `detail::validate`, `detail::validate_preselected_argmax`, `detail::bind_inputs`,
  `detail::extract_outputs`, and `detail::extract_preselected_argmax` so their graph callback
  outputs cannot produce invalid/backend graph outcomes on the maintained generator path.
- Strengthen source scans to cover the full maintained callback span from validate through extract,
  not only kernel wrapper spans.
- Preserve graph processor generic behavior; Phase 147 is about the generator maintained path and
  its action-called detail callbacks.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>

---

*Phase: 147-text-generator-graph-callback-outcome-ownership-closure*
*Context gathered: 2026-04-30 via autonomous gap closure*
