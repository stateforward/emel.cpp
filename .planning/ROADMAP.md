# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](.planning/milestones/v1.0-ROADMAP.md)
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](.planning/milestones/v1.1-ROADMAP.md)
- [x] [v1.2: Flash Attention](.planning/milestones/v1.2-ROADMAP.md)
- [x] [v1.3: ARM Flash Optimizations](.planning/milestones/v1.3-ROADMAP.md)
- [x] [v1.4: Full Vectorized Quantized Kernels](.planning/milestones/v1.4-ROADMAP.md)
- [x] [v1.5: Full ARM Quantized Path](.planning/milestones/v1.5-ROADMAP.md)
- [x] [v1.6: Qwen3-0.6B Parity And Benchmark](.planning/milestones/v1.6-ROADMAP.md)
- [x] [v1.7: Generator Prefill Submachine Decomposition](.planning/milestones/v1.7-ROADMAP.md)
- [x] [v1.8: Truthful Qwen3 E2E Embedded Size](.planning/milestones/v1.8-ROADMAP.md)
- [x] [v1.9: Liquid LFM2.5-1.2B Thinking ARM Slice](.planning/milestones/v1.9-ROADMAP.md)
- [x] [v1.11: TE-75M GGUF Trimodal Embedding Runtime](.planning/milestones/v1.11-ROADMAP.md)
  - Shipped 2026-04-15 with maintained TE trimodal embedding runtime support, refreshed closeout
    evidence, and a passing milestone audit.
- [x] [v1.12: Pluggable Reference Parity Bench Architecture](.planning/milestones/v1.12-ROADMAP.md)
  - Shipped 2026-04-18, reopened narrowly for archived closeout-proof repair on 2026-04-19, and
    returned to a passing rerun audit on 2026-04-20.
- [x] [v1.13: Pluggable Generative Parity Bench](.planning/milestones/v1.13-ROADMAP.md)
  - Shipped 2026-04-21 with a maintained generative compare contract, workload manifests,
    `llama_cpp_generation` reference lane, truthful comparable/non-comparable publication, and a
    no-blocker audit.
- [x] [v1.14: Benchmark Variant Organization](.planning/milestones/v1.14-ROADMAP.md)
  - Shipped 2026-04-21 with deterministic data-owned benchmark variant discovery for generation
    and embedding variants.
- [x] [v1.15: ARM Sortformer Diarization GGUF Slice](.planning/milestones/v1.15-ROADMAP.md)
  - Shipped 2026-04-25 with one maintained native Sortformer diarization GGUF slice, PyTorch/NeMo
    parity, ONNX CPU single-thread benchmark reference, EMEL-over-ONNX performance closure, and a
    passing source-backed milestone audit.
- [x] [v1.16: ARM Whisper GGUF Parity And Performance](.planning/milestones/v1.16-ROADMAP.md)
  - Shipped 2026-04-28 with one maintained Whisper tiny GGUF ASR slice, speech-owned runtime
    actors, recognizer-backed exact transcript parity, matched single-thread ARM benchmark proof,
    and source-backed closeout evidence.
- [x] [v1.17: Text Generator Domain Alignment](.planning/milestones/v1.17-ROADMAP.md)
  - Closed 2026-04-30 after Phase 147 removed the final source-backed `TEXTGEN-04` /
    `TEXTGEN-07` blocker: maintained graph validation, bind, and extract callbacks no longer
    route graph outcomes through action-called `detail.hpp` helper failures or `err_out`.

## Current Milestone

v1.17 is ready to complete after Phase 147. Phases 144 and 145 removed hidden route-selection
paths, Phase 146 closed the audited run-kernel wrapper portion, and Phase 147 removed the remaining
graph validation/bind/extract callback outcome bypass. The required full closeout quality gate and
final source-backed milestone audit both passed.

- [x] **Phase 144: Text Generator Runtime Route Ownership Closure** - Close the remaining
  source-backed `TEXTGEN-04` / `TEXTGEN-07` runtime route ownership gaps.
- [x] **Phase 145: Text Generator Native Quantized Route Evidence Closure** - Close the
  remaining `native_quantized` route ownership, regression coverage, lifecycle selection, and
  scoped evidence gaps from the refreshed milestone audit.
- [x] **Phase 146: Text Generator Explicit Compute Outcome Modeling Closure** - Move audited
  graph compute validation and success/error outcome decisions out of action-called
  `detail.hpp` wrappers and into explicit Boost.SML behavior.
- [x] **Phase 147: Text Generator Graph Callback Outcome Ownership Closure** - Close the
  remaining maintained graph validation, bind, and extract callback outcome bypass by moving
  outcome decisions out of action-called `detail.hpp` helper outputs and into explicit SML
  guards/transitions.

## Phase Details

### Phase 144: Text Generator Runtime Route Ownership Closure

**Goal:** Close the remaining source-backed `TEXTGEN-04` / `TEXTGEN-07` blocker by moving
maintained generator runtime path choices out of action-called `detail.hpp` helpers and into
explicit guard-owned predicates plus `sm.hpp` transitions.

**Requirements:** `TEXTGEN-04`, `TEXTGEN-07`

**Gap Closure:** Closes `.planning/milestones/v1.17-MILESTONE-AUDIT.md` findings for:

- matmul route choice currently inside `detail::matmul_vector(...)`
- prefill/decode choice currently inside `detail::run_kernel_mode(...)` and
  `detail::run_kernel_mode_preselected_argmax(...)`
- maintained parity/benchmark evidence proving the explicit routes are the paths being run

**Success Criteria**:
1. Maintained generator compute no longer chooses matmul route variants inside action-called
   `detail.hpp` helpers; route support/readiness choices are guard-owned and visible in SML
   transitions.
2. Maintained prefill/decode kernel execution no longer chooses phase kind inside action-called
   `detail.hpp` helpers; prefill/decode routes are explicit in parent/prefill SML graphs.
3. Source regressions fail on reintroduced audited detail-owned route selection for
   `matmul_vector`, `run_kernel_mode`, and `run_kernel_mode_preselected_argmax`.
4. Existing generation behavior, fixture scope, sampling policy, public events, and EMEL/reference
   lane isolation remain unchanged.
5. Focused generator/runtime tests, domain-boundary checks, scoped generation quality gate,
   paritychecker evidence, and generation benchmark evidence pass. Models, snapshots, and
   benchmark baselines may be updated when needed for truthful maintained-path proof.

**Status:** Superseded. Phase 144 removed the first route-selection blockers; Phase 145 repaired
native-quantized route evidence; Phase 146 closed the audited run-kernel wrapper portion. Phase
147 owns the final validation/bind/extract callback outcome closure for `TEXTGEN-04` /
`TEXTGEN-07`.

### Phase 145: Text Generator Native Quantized Route Evidence Closure

**Goal:** Close the refreshed source-backed `TEXTGEN-04` / `TEXTGEN-07` blockers by moving
the remaining maintained `native_quantized` packed-q8 / q8-k / kernel fallback choice out of
action-called `detail.hpp`, covering the missed regression path, resolving the lifecycle selection
concern, and producing passing scoped generation evidence.

**Requirements:** `TEXTGEN-04`, `TEXTGEN-07`

**Gap Closure:** Closes `.planning/milestones/v1.17-MILESTONE-AUDIT.md` findings for:

- `detail::matmul_vector_native_quantized(...)` choosing packed-q8, q8-k, or kernel fallback at
  runtime.
- `run_layer<..., native_quantized>` and `compute_logits<native_quantized>` consuming that hidden
  route choice from action-called detail code.
- Phase 144 source regression missing the native-quantized helper where the remaining route choice
  lives.
- `detail::phase_lifecycle(...)` using an action-called runtime-indexed lifecycle manifest.
- Phase 144 scoped quality gate coverage reporting failure.

**Success Criteria**:
1. Maintained `native_quantized` matmul route selection is represented by guard-owned predicates
   and explicit parent/prefill SML transitions rather than action-called `detail.hpp` branching.
2. Source regressions fail on reintroduced native-quantized route selection in
   `detail::matmul_vector_native_quantized(...)` and continue to fail on the removed generic
   `run_kernel_mode` wrappers.
3. The action-called `phase_lifecycle(...)` runtime-indexed manifest choice is either moved into
   explicit route-owned construction or documented with a source-backed rule justification that
   does not affect runtime behavior selection.
4. Existing generation behavior, fixture scope, sampling policy, public events, and EMEL/reference
   lane isolation remain unchanged.
5. Focused generator/runtime tests, domain-boundary checks, paritychecker evidence, scoped
   generation quality gate, and generation benchmark evidence pass. Model, benchmark, and snapshot
   updates are explicitly permitted when needed for truthful maintained-path proof.

**Status:** Complete.

### Phase 146: Text Generator Explicit Compute Outcome Modeling Closure

**Goal:** Close the remaining source-backed `TEXTGEN-04` / `TEXTGEN-07` blockers by moving graph
compute validation outcomes and success/error behavior out of action-called `detail.hpp`
run-kernel wrappers and into explicit guard-owned predicates, states, and error transitions.

**Requirements:** `TEXTGEN-04`, `TEXTGEN-07`

**Gap Closure:** Closes `.planning/milestones/v1.17-MILESTONE-AUDIT.md` findings for:

- `detail::run_kernel_scalar_mode(...)`,
  `detail::run_kernel_scalar_preselected_argmax_mode(...)`, and
  `detail::run_kernel_prefill_chunk8_q8_k_mode(...)` branching on runtime request plan, step kind,
  backend readiness, bound state, output pointers, token counts, chunk readiness, and kernel
  result to decide `err_out` and the next success/error path.
- `action::request_phase_compute<...>` binding those helpers into dispatch such that helper
  return values and `err_out` decide downstream graph compute outcome behavior.
- Phase 144/145 closeout evidence claiming completion while roadmap state, summary frontmatter,
  and live source still contradict explicit behavior modeling readiness.

**Success Criteria**:
1. All graph compute preconditions that decide acceptance, invalid request, backend error, missing
   output, wrong step kind, chunk readiness, or selected route availability are represented by
   explicit guards and destination-first SML transition rows in the parent and/or prefill actors.
2. Action-called `detail.hpp` graph compute callbacks execute only an already-selected numeric
   operation or data-plane kernel path; they do not choose validation outcomes, fallback routes,
   callback/error channels, or "what happens next" through runtime branching, `err_out`, or helper
   return values.
3. Source regressions fail on reintroduced action-called detail validation/outcome branching in the
   audited run-kernel wrappers and on missing Phase 146 requirement frontmatter / roadmap
   traceability.
4. Existing generation behavior, fixture scope, sampling policy, public events, EMEL/reference lane
   isolation, and maintained parity/benchmark contracts remain unchanged except for explicitly
   approved model, snapshot, or benchmark updates needed to prove the real maintained path.
5. Focused generator/runtime tests, domain-boundary checks, paritychecker evidence, full
   bench-runner completion for generation-relevant lanes, scoped generation quality gate, and
   generation benchmark evidence pass before another milestone audit is attempted.

**Status:** Superseded by Phase 147 for final closeout. Phase 146 validation passed focused
generator tests, domain-boundary checks, paritychecker, generation bench-runner smoke, the scoped
generation quality gate, and generation benchmark comparison for the audited run-kernel wrapper
path. The refreshed milestone audit then found the maintained graph validation/bind/extract
callback outcome path still routed through action-called detail helper outputs, so Phase 147 owns
the final source-backed gap closure.

### Phase 147: Text Generator Graph Callback Outcome Ownership Closure

**Goal:** Close the remaining source-backed `TEXTGEN-04` / `TEXTGEN-07` blocker by ensuring the
maintained text generator graph validation, bind, and extract outcome decisions are explicit in
Boost.SML guards/transitions rather than hidden in action-called `detail.hpp` callbacks.

**Requirements:** `TEXTGEN-04`, `TEXTGEN-07`

**Gap Closure:** Closes `.planning/v1.17-MILESTONE-AUDIT.md` findings for:

- `src/emel/text/generator/actions.hpp` wiring `detail::validate`,
  `detail::validate_preselected_argmax`, `detail::bind_inputs`, `detail::extract_outputs`, and
  `detail::extract_preselected_argmax` into graph compute callbacks.
- `src/emel/text/generator/detail.hpp` callbacks returning `bool` and writing `err_out` for
  validation, bind, or extract outcome decisions.
- `src/emel/graph/processor/*_step/actions.hpp` and guards routing done/error transitions from
  those callback outputs on the maintained generator path.
- Previously stale closeout docs that still published the obsolete Phase 143 closeout story while
  the active milestone was reopened through Phase 147.

**Allowed updates:** model files, snapshots, benchmark baselines, and generation benchmark
fixtures may be updated when required for truthful maintained-path proof. These updates are not a
substitute for removing the callback outcome bypass.

**Required Tasks:**
1. Move materialized and preselected graph compute validation readiness for step plan presence,
   expected output count, compute I/O, logits/selected-output pointers, positions, token counts,
   backend readiness, bound capacity, and extract readiness into explicit generator/prefill guards
   and destination-first SML transition rows before graph compute dispatch.
2. Refactor the maintained generator graph callbacks so validation, bind, and extract detail
   helpers perform only already-accepted data binding or output copying; helper return values and
   `err_out` must not decide graph validation, callback, error-channel, or done/error behavior.
3. Add source and actor-level regressions that fail if the maintained path again routes
   validation, bind, or extract outcomes through `text/generator/detail.hpp` callback outputs.
4. Prove paritychecker, generation benchmark, and embedded-size probe still construct
   `emel::text::generator::sm` and drive public `process_event(...)` without actor-internal
   detail/action/guard bridges.
5. Repair `.planning/MILESTONES.md`, `.planning/PROJECT.md`, `.planning/STATE.md`, active
   roadmap/requirements, and milestone archive copies so all closeout evidence names Phase 147 as
   the final reopened gap closure.

**Success Criteria**:
1. No action-called `text/generator/detail.hpp` helper output decides graph validation, bind,
   extract, callback, or error-channel behavior for the maintained generator path.
2. Parent and/or prefill SML guards explicitly classify every malformed request, backend,
   expected-output, position, token-count, and output-pointer condition formerly handled by detail
   validation/extract callbacks.
3. Source regressions scan the maintained callback span and fail on reintroduced `request_plan`,
   `check_backend`, output-pointer branching, `expected_outputs`, `positions`, token counts,
   `*err_out`, or `k_error_invalid` outcome routing in action-called detail callbacks.
4. Focused generator/runtime tests, graph callback tests, domain-boundary checks, paritychecker
   evidence, generation benchmark evidence, and the scoped generation quality gate pass.
5. Any model, snapshot, or benchmark updates are explicitly documented as maintained-path proof
   updates, and no fallback kernel, tool-only scaffold, or synthetic fixture is substituted for the
   intended runtime path.

**Status:** Complete. Phase 147 refactored the maintained generator graph validation/bind/extract
callbacks into guard-accepted callbacks that cannot reject graph execution or write graph callback
error state through `err_out`. Source regressions now scan the maintained callback spans and action
wiring. Focused generator/text/kernel graph CTest slices, lint snapshot, domain-boundary checks,
paritychecker, generation benchmark evidence, the changed-file scoped quality gate, and the required
full closeout quality gate passed.
