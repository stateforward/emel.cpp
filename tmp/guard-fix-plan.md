# Guard Fix Plan (Precondition Checks)

Scope
- Convert action-level precondition/early-return checks into guards + explicit transitions.
- Preserve current error codes and outcome events.
- Avoid any action-side branching for preconditions.

Approval requirement
- Per AGENTS.md, I will ask before changing any state machine structure.

Audit findings (by machine)

Buffer planner
- Validation of `buffer_count`, graph node/leaf counts, null pointers, tensor count bounds.
- Output capacity checks (`sizes_out`, `chunk_counts_out`).
- Alignment/max-size array validation.
- Owner dispatch null checks and strategy pointer checks.
- Strategy function pointer null checks (`seed_leafs`, `count_references`, `alloc_explicit_inputs`, `plan_nodes`, `release_expired`, `finalize`).

Buffer allocator
- Planner/analyzer/chunk allocator null checks.
- Graph validity and buffer id validation checks.
- Output capacity checks.

Buffer chunk allocator
- Alignment/max-size validation.
- Request validation for allocation/release paths.

Buffer realloc analyzer
- Graph validity and alloc array checks.

Model loader
- Map parser / dispatch / parser / weight loader pointer checks.
- Structure/architecture validation preconditions.

Model parser
- Parse/map callbacks and tensor mapping preconditions.
- Owner dispatch preconditions.

Model weight loader
- Request and callback preconditions (mmap/stream/validate/cleanup).
- Owner dispatch preconditions.

Progress
- Decoder: moved ubatch-size precondition to guards; actions no longer early-return for invalid sizes.
- Buffer planner: moved request-null preconditions into guards + error transitions; removed strategy checks from
  run helpers; begin_plan no longer performs validation. Tests updated. Gates green.
- Buffer allocator: moved reserve/alloc/initialize preconditions into guards and explicit reject transitions;
  removed planner/analyzer/chunk allocator null preconditions in action helpers. Tests updated.

Current status
- Next up: model loader / parser / weight_loader guard migration.

Proposed approach
1. For each machine, define guard predicates covering the preconditions currently enforced in actions.
2. Add explicit transitions for invalid-precondition paths to dedicated error events/states, reusing existing `_error` events where possible.
3. Keep side effects in actions only; guards remain pure.
4. Update tests for any behavior changes triggered by explicit guard transitions.

Suggested order (highest ROI / cross-cutting)
1. `src/emel/buffer/planner/*` (many shared checks, feeds allocator).
2. `src/emel/buffer/allocator/*` (hot path and consumer of planner).
3. `src/emel/model/loader/*` + `src/emel/model/parser/*` + `src/emel/model/weight_loader/*`.
4. `src/emel/buffer/chunk_allocator/*`.
5. `src/emel/buffer/realloc_analyzer/*`.

Open decisions
- Whether to introduce shared validation guards per machine or keep them in per-event guard structures.
- Whether to add explicit `*_invalid` events vs reusing existing `_error` events with `EMEL_ERR_INVALID_ARGUMENT`.

- model/loader: moved request validation into guards (map_parser/parse/load_weights/map_layers/structure/architecture), added explicit invalid transitions, and updated loader action tests to assert guards.
