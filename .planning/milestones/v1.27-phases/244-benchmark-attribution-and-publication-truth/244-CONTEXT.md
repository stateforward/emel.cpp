# Phase 244: Benchmark Attribution and Publication Truth - Context

**Gathered:** 2026-06-25
**Status:** Ready for planning
**Mode:** Auto-generated (autonomous processor-support phase)

<domain>
## Phase Boundary

Publish the maintained benchmark and documentation evidence for the Ryzen
AVX2/FMA work from Phases 239-243. This phase does not add new kernels or widen
the runtime contract. It closes the truthfulness loop by making the maintained
benchmark snapshot and generation publication baselines match the source-backed
x86_64 optimized path evidence, while keeping unsupported feature families as
explicit no-claims.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Surface
- Use the existing `tools/bench` maintained `kernel_x86_64` suite. It is wired
  through `tools/bench/bench_runner_registry.cpp`,
  `tools/bench/kernel/x86_64_bench.cpp`, and
  `scripts/quality_gates.sh`.
- The benchmark snapshot baseline is `snapshots/bench/benchmarks.txt`.
- Do not update benchmark snapshots without explicit user approval.

### Generation Publication Surface
- Maintained generation publication baselines live under `snapshots/parity/`.
- Phase 243 proved live EMEL/reference generation matches for `1`, `10`, `100`,
  and `1000` token runs. The `10`, `100`, and `1000` publication baselines are
  stale and need explicit approval before update.
- Use paritychecker's existing `--write-generation-baseline <path>` support for
  baseline writes. Do not rewrite parity baselines without explicit approval.

### Truthfulness Rules
- Published output must identify this host as AMD Ryzen 9 5950X with x86_64
  AVX2, FMA, and F16C conversion support only.
- Published output must not imply AVX-512, AVX-VNNI, AMX, BF16, native FP16, GPU,
  or llama.cpp/ggml runtime acceleration.
- Benchmark and parity lanes must remain separated: EMEL-owned code produces the
  EMEL result, and llama.cpp/ggml remains comparison-only on the reference side.

### Validation
- Run `scripts/bench.sh --snapshot --compare --suite=kernel_x86_64` before any
  approved update to confirm the current missing baseline set.
- After explicit approval, run
  `scripts/bench.sh --snapshot --update --suite=kernel_x86_64`.
- After explicit approval, refresh stale maintained generation baselines for the
  live-matching `10`, `100`, and `1000` token runs.
- Re-run the changed-file scoped quality gate with
  `EMEL_QUALITY_GATES_BENCH_SUITE=kernel_x86_64`.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/kernel/x86_64_bench.cpp` appends EMEL and reference
  `kernel_x86_64` benchmark cases.
- `tools/bench/bench_runner_registry.cpp` registers the `kernel_x86_64` suite.
- `tools/bench/bench_dependency_manifest.cpp` maps `kernel_x86_64` to
  `tools/bench/kernel/x86_64_bench.cpp`, `tools/bench/kernel/bench_common.hpp`,
  and `src/emel/kernel`.
- `scripts/bench.sh` supports suite-scoped snapshot updates and merges them into
  `snapshots/bench/benchmarks.txt`.
- `tools/paritychecker/parity_runner.cpp` supports
  `--write-generation-baseline`.
- `tools/paritychecker/parity_engines.cpp` computes the default maintained
  generation baseline path under `snapshots/parity/`.

### Integration Points
- `snapshots/bench/benchmarks.txt`: approved `kernel_x86_64` snapshot entries.
- `snapshots/parity/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10.txt`
- `snapshots/parity/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100.txt`
- `snapshots/parity/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000.txt`
- `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, and
  `.planning/PROJECT.md` for final traceability after approved snapshot updates.

</code_context>

<specifics>
## Specific Ideas

- Treat `scripts/bench.sh --snapshot --compare --suite=kernel_x86_64` as the
  publication preflight; it should fail only because the maintained snapshot
  lacks the new `kernel/x86_64/*` entries.
- Treat the generation baseline updates as publication baselines, not runtime
  proof. Phase 243 already proved live EMEL/reference output equality.
- Final closeout requires a clean scoped quality gate, not just artifact edits.

</specifics>

<active_next_scope>
## Active Next Scope

- Get explicit snapshot approval, run the approved updates, rerun the scoped
  quality gate, then mark `XBN-01` and `XBN-02` source/gate complete.

</active_next_scope>

---

*Phase: 244-benchmark-attribution-and-publication-truth*
*Context gathered: 2026-06-25*
