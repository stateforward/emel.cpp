# Phase 154: Generation Live Reference Truth Closure - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Close `PARITY-03` and `LANE-01` by making maintained generation parity compare EMEL output against
live reference-lane generation output on the normal success path.

</domain>

<decisions>
## Implementation Decisions

### Live Reference Truth
- Keep the maintained generation entrypoint in `tools/paritychecker/parity_engines.cpp`.
- Run `run_reference_generate(state.reference, opts, reference_result)` after the EMEL generation
  result and diagnostics are available.
- Compare EMEL output and trace data against the live reference result, not against
  `baseline_record.result`.
- Keep `load_generation_reference_backend(opts.model_path, state)` as the separate reference-lane
  model/vocab owner.

### Snapshot Baselines
- Treat stored generation baselines as append-only publication artifacts.
- On normal non-write runs, load the baseline after the live comparison and check it against the
  live reference result to detect stale publication artifacts.
- On `--write-generation-baseline`, write the live reference result only after EMEL/live-reference
  parity has passed.
- User has explicitly allowed snapshot/model updates if they are required by this closure phase.

### Verification Shape
- Add a source guard that fails if the harness goes back to aliasing
  `baseline_record.result` as the reference truth.
- Strengthen the maintained generation runtime smoke to require live reference decode calls and
  stable live-reference/baseline output markers.
- Preserve deterministic missing-model behavior and existing fixture restrictions.
- Run focused paritychecker tests and changed-file scoped quality gates.

</decisions>

<code_context>
## Existing Code Insights

### Current Gap
- `run_generation_harness_contract(...)` loads a live reference backend, but the success comparison
  binds `generation_result & reference_result = baseline_record.result`.
- `run_reference_generate(...)` already exists and can produce a `generation_result` from the
  separate reference backend.
- `dump_reference_decode_seam(...)` already publishes reference decode call counts and reference
  source/ref metadata.

### Relevant Tests
- `paritychecker matches maintained generation baselines across supported fixtures` already drives
  the generation paritychecker against maintained fixtures when the models are present.
- `parity engine source keeps EMEL and reference lane objects separate` already checks lane-owned
  object construction and should be extended for live reference truth.
- `paritychecker generation reports a deterministic missing-model failure` covers missing-model
  failure stability.

</code_context>

<specifics>
## Specific Ideas

Rename the stored-baseline output line to `generation_baseline:` so the `reference_impl:` line
continues to mean the live reference implementation rather than the snapshot artifact.

</specifics>

<deferred>
## Deferred Ideas

Actor helper boundary enforcement is owned by Phase 155. Dependency manifest gate consumption is
owned by Phase 156.

</deferred>
