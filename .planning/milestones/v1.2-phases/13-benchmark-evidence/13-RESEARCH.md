# Phase 13: Benchmark Evidence - Research

**Researched:** 2026-03-22
**Domain:** truthful flash-attention benchmark publication for the canonical Llama-68M generation
slice
**Confidence:** MEDIUM

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BENCH-01 | `tools/bench` can run the canonical flash-attention path through the existing compare workflow on the same maintained workload contract. | `tools/bench/generation_bench.cpp` already runs the canonical `hello` / `max_tokens=1` and `max_tokens=8` generation cases through the shipped EMEL generator and the explicit `llama.cpp` reference path. The remaining work is to make the compare workflow publishable only when the EMEL case is proven to have executed flash attention and when the reference source is deterministic. |
| BENCH-02 | Benchmark output and maintained artifacts clearly distinguish flash-attention evidence from the previous non-flash baseline. | The current compare/docs surfaces overwrite the old generic generation row and do not preserve the prior non-flash baseline anywhere in maintained artifacts. Git history still contains the pre-flash short-case baseline, so the phase can publish separate flash evidence without inventing a benchmark-only runtime path if it preserves that baseline as a first-class artifact. |
| BENCH-03 | Canonical flash-attention benchmark results show measurable improvement over the current EMEL non-flash canonical baseline on at least one maintained case. | The durable pre-flash short-case compare snapshot in git history was `63,837,917 ns/op`, while the current snapshot is `6,992,875 ns/op`, a `9.129x` speedup and `89.0%` latency drop on the maintained `max_tokens=1` case. The long `max_tokens=8` case was added later, so the short case is the safest mandatory maintained improvement proof. |
</phase_requirements>

## Summary

Phase 13 is not a new benchmark-harness phase. The benchmark runtime already exists in
`tools/bench`: `bench_runner --mode=compare` runs both EMEL and `llama.cpp`, the canonical
generation workload is already fixed to `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` with prompt
`hello`, and the maintained artifact chain already flows through `scripts/bench.sh`,
`snapshots/bench/benchmarks_compare.txt`, `snapshots/bench/benchmarks.txt`, and
`tools/docsgen/docsgen.cpp` into `docs/benchmarks.md`.

The real gap is truthfulness and artifact shape. Today the benchmark workflow publishes only a
generic generation row, while flash proof still lives in paritychecker and in the opt-in
`EMEL_BENCH_AUDIT_GENERATION_SEAMS` stderr path. That means the benchmark can produce numbers
without a durable published proof that the EMEL side actually executed the shipped flash path. It
also means the current artifact flow overwrites the old non-flash evidence instead of preserving it
as a maintained baseline.

There is also a concrete workflow risk in the current repo state: unlike paritychecker,
`tools/bench/CMakeLists.txt` still prefers a local `tmp/llama.cpp` checkout over the configured
reference ref. On this machine, a low-iteration `scripts/bench.sh --compare` attempt on
2026-03-22 failed during CMake configuration after selecting `tmp/llama.cpp`, so Phase 13 should
assume the benchmark workflow is not yet truthfully reproducible until that reference-selection
policy is fixed or explicitly justified.

**Primary recommendation:** keep one benchmark execution surface
(`bench_runner --mode=compare` / `scripts/bench.sh --compare`), but make published flash evidence
depend on three things before artifact refresh: deterministic reference sourcing, durable flash
execution proof, and an explicit preserved non-flash baseline artifact for the short canonical
generation case.

## Project Constraints (from AGENTS.md)

- Keep the phase inside the existing `tools/bench`, compare, snapshot, and docsgen surfaces.
- Do not widen public API or runtime scope to publish benchmark evidence.
- Keep benchmark claims truthful to the shipped flash path; do not publish numbers from a tool-only
  shortcut or a non-equivalent runtime path.
- `tools/bench` and `tools/paritychecker` are the only approved places that may link EMEL with
  `llama.cpp` / `ggml`.
- Snapshot or docs baseline refresh remains an explicit user-approved action; do not plan implicit
  snapshot churn.
- `scripts/quality_gates.sh` is still required after implementation changes, but its current
  warning-only benchmark policy must not be mistaken for proof that Phase 13 is complete.
- Preserve current Boost.SML / generator / graph runtime boundaries; benchmark publication is a
  tool-surface phase, not a state-machine redesign.

## Standard Stack

### Core
| Library / Component | Version | Purpose | Why Standard |
|---------------------|---------|---------|--------------|
| `tools/bench/bench_main.cpp` | repo local | Authoritative compare runner and printed row format | This is already the only maintained compare surface for EMEL vs `llama.cpp` publication. |
| `tools/bench/generation_bench.cpp` | repo local | Canonical Llama-68M generation benchmark implementation | It already reuses the shipped EMEL generator path and the explicit reference path for the maintained workload contract. |
| `scripts/bench.sh` | repo local | Operator entrypoint for `--compare`, `--compare-update`, and `--snapshot --compare` | This script already drives the compare/artifact workflow and should remain the only operator-facing publish path. |
| `snapshots/bench/benchmarks_compare.txt` | current ref `ecbcb7ea9d3303097519723b264a8b5f1e977028` | Durable compare snapshot used for docs publication | `tools/docsgen` reads only this file today, so published benchmark docs flow from it. |
| `snapshots/bench/benchmarks.txt` | current ref `ecbcb7ea9d3303097519723b264a8b5f1e977028` | Durable EMEL snapshot baseline used by benchmark gating | This remains the regression-gate baseline for the EMEL side. |
| `tools/docsgen/docsgen.cpp` | repo local | Generated benchmark-docs pipeline | The current benchmark docs are derived from the compare snapshot here, not hand-maintained. |

### Supporting
| Library / Component | Version | Purpose | When to Use |
|---------------------|---------|---------|-------------|
| `tools/paritychecker/parity_runner.cpp` | repo local | Existing durable flash-execution proof surface | Use as the truth reference for how flash proof should be published by tools without widening runtime/API scope. |
| `tools/paritychecker/paritychecker_tests.cpp` | repo local | Existing subprocess coverage for flash proof | Use as the pattern for shell-visible proof assertions when adding benchmark publication checks. |
| `tools/bench/reference_ref.txt` | `ecbcb7ea9d3303097519723b264a8b5f1e977028` | Reproducible benchmark reference pin | Keep the pin if reproducibility remains the benchmark policy, but do not let a local `tmp/llama.cpp` checkout silently override it. |
| Git history for `snapshots/bench/benchmarks*.txt` | commit `2acd4fe^` is the last pre-flash generation baseline | Durable non-flash baseline source | Use to seed a maintained baseline artifact instead of retyping historical numbers. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Existing compare flow plus stronger artifacts | A new flash-only benchmark runner or script | Faster to prototype, but it violates the milestone boundary and creates two truths for the same workload. |
| Preserved non-flash baseline artifact in `snapshots/bench` / generated docs | Keeping the old baseline only in git history | Cheaper short-term, but BENCH-02 becomes weak because the maintained artifact flow no longer distinguishes baseline from flash evidence. |
| Deterministic fetched-or-pinned reference source | Preferring local `tmp/llama.cpp` when present | Convenient for local hacking, but benchmark evidence becomes machine-dependent and failed on this machine during research. |

**Installation:** not applicable. Phase 13 should use the repo’s existing `cmake`, `ninja`, `zig`,
`scripts/bench.sh`, and `scripts/generate_docs.sh` tooling rather than introducing new packages.

## Architecture Patterns

### Recommended Project Structure

```text
tools/
├── bench/                     # runtime benchmark cases and compare runner
├── docsgen/                   # generated benchmark docs from snapshot artifacts
└── paritychecker/             # existing flash-proof publication reference
scripts/
├── bench.sh                   # operator entrypoint for compare/snapshot refresh
└── generate_docs.sh           # docs artifact generation
snapshots/
└── bench/                     # maintained benchmark evidence artifacts
docs/
├── benchmarking.md            # operator runbook
└── benchmarks.md              # generated published evidence
```

### Pattern 1: Keep One Execution Surface
**What:** Publish flash evidence only through the existing `bench_runner --mode=compare` and
`scripts/bench.sh --compare` path.

**When to use:** For all canonical generation benchmark publication work in Phase 13.

**Example:**

```bash
# Source: scripts/bench.sh + tools/bench/bench_main.cpp
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --compare
```

### Pattern 2: Publish Flash Proof As Artifact Metadata, Not As A New Runtime Path
**What:** Reuse the existing generator flash counters and seam-audit truth from the shipped runtime,
but make the benchmark publication workflow refuse to bless numbers unless the canonical EMEL case
proves flash execution.

**When to use:** When refreshing compare snapshots or generating docs for flash evidence.

**Example:**

```bash
# Source: tools/bench/generation_bench.cpp + src/emel/generator/sm.hpp
EMEL_BENCH_AUDIT_GENERATION_SEAMS=1 \
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --compare 2>&1 | \
  rg 'generation_bench_seams|generation/preloaded_request/'
```

### Pattern 3: Preserve Historical Non-Flash Baseline As A First-Class Artifact
**What:** BENCH-02 and BENCH-03 are best satisfied by materializing the pre-flash short-case
baseline from committed snapshot history into a maintained artifact or generated doc section rather
than relying on git archaeology at verification time.

**When to use:** Before the first approved flash snapshot refresh.

**Example:**

```bash
# Source: git history of snapshots/bench/benchmarks_compare.txt
git show 2acd4fe^:snapshots/bench/benchmarks_compare.txt | \
  rg 'generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1'
```

### Pattern 4: Reference Source Must Be Deterministic And Non-Local
**What:** `tools/bench` should follow a deterministic fetched-or-pinned reference policy, not a
best-effort local `tmp/llama.cpp` override.

**When to use:** Before trusting any newly published benchmark evidence.

**Example:**

```cmake
# Source: tools/paritychecker/CMakeLists.txt
FetchContent_Declare(
  reference_impl
  GIT_REPOSITORY ${REF_IMPL_REPOSITORY}
  GIT_TAG ${REF_IMPL_REF}
)
```

### Anti-Patterns to Avoid

- **Benchmark-only flash path:** Do not add a special benchmark runtime that bypasses the shipped
  generator -> graph -> processor -> kernel chain.
- **Optional truth gating:** Do not leave flash proof as a stderr-only local audit if Phase 13 is
  supposed to publish durable flash evidence.
- **Overwriting the only baseline:** Do not refresh compare/docs artifacts in a way that erases the
  pre-flash short-case baseline from maintained repo surfaces.
- **Machine-dependent reference builds:** Do not keep `tools/bench` preferring `tmp/llama.cpp`
  when the milestone requires truthful published evidence.
- **Ad hoc docs edits:** Do not hand-edit `docs/benchmarks.md`; continue generating it from
  maintained snapshot artifacts.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Flash benchmark publication | A new benchmark runner or a flash-only CLI | `bench_runner --mode=compare` plus `scripts/bench.sh --compare` | The compare flow already exists and is the accepted artifact surface. |
| Flash truth proof | Manual reasoning from performance deltas | Existing generator flash counters and bench seam-audit signals | Numbers alone do not prove the flash path ran. |
| Non-flash baseline evidence | Hand-entered markdown tables or planning-only notes | A maintained snapshot artifact derived from committed benchmark history | BENCH-02 needs durable repo evidence, not ephemeral planning notes. |
| Published docs refresh | Manual editing of `docs/benchmarks.md` | `scripts/generate_docs.sh` / `tools/docsgen/docsgen.cpp` | The docs pipeline is already generated and deterministic. |
| Reference source selection | A local `tmp/llama.cpp` preference | Deterministic CMake fetch/pin using `REF_IMPL_REF` | Machine-local overrides make benchmark evidence untrustworthy and already failed during research. |

**Key insight:** Phase 13 should add evidence and verification around the existing benchmark path,
not create a second path that happens to be easier to label as “flash”.

## Common Pitfalls

### Pitfall 1: Publishing Generic Generation Rows As If They Were Flash Evidence
**What goes wrong:** The compare/docs artifacts still look like generic generation numbers even
though the milestone claims flash-attention evidence.

**Why it happens:** `tools/bench/bench_main.cpp`, `snapshots/bench/benchmarks_compare.txt`, and
`tools/docsgen/docsgen.cpp` currently treat the canonical generation rows as plain generation rows
with no persisted flash-specific metadata.

**How to avoid:** Plan an artifact-level distinction for flash evidence before refreshing
snapshots. The planner should decide early whether that distinction is encoded in case names,
sidecar snapshot metadata, or generated-doc sections.

**Warning signs:** `docs/benchmarks.md` still renders only the generic
`generation/preloaded_request/...` rows and nothing in maintained artifacts points to the old
non-flash baseline.

### Pitfall 2: Trusting Bench Output Without Durable Flash-Execution Proof
**What goes wrong:** Phase 13 publishes faster numbers, but the maintained artifacts do not prove
the EMEL side actually executed the shipped flash path.

**Why it happens:** `generation_bench.cpp` can optionally print seam-audit output, but normal
compare rows do not encode flash dispatch counts today.

**How to avoid:** Require a publish-time proof step that consumes existing generator counters or
seam-audit truth before snapshot/doc refresh is considered valid.

**Warning signs:** The workflow can update `benchmarks_compare.txt` even when no artifact records
flash dispatch or seam separation.

### Pitfall 3: Letting `tmp/llama.cpp` Decide The Benchmark Reference
**What goes wrong:** Benchmark evidence changes depending on a developer’s local checkout or fails
to build entirely.

**Why it happens:** `tools/bench/CMakeLists.txt` still prefers `tmp/llama.cpp` over the configured
`REF_IMPL_REF`, unlike paritychecker’s Phase 12 policy.

**How to avoid:** Remove or explicitly gate the local override and make benchmark builds consume the
configured reference pin deterministically.

**Warning signs:** CMake output mentions `tmp/llama.cpp`, or `scripts/bench.sh --compare` fails on
one machine but not another before the benchmark even starts.

### Pitfall 4: Losing The Pre-Flash Baseline During Artifact Refresh
**What goes wrong:** BENCH-03 becomes impossible to prove from maintained artifacts because the
old non-flash number is only recoverable from git history.

**Why it happens:** `--compare-update` overwrites the compare snapshot and `docsgen` renders only
the current file.

**How to avoid:** Preserve the pre-flash short-case baseline as an intentional maintained artifact
before refreshing flash-labeled evidence.

**Warning signs:** The only way to answer “what was the old non-flash number?” is `git show`.

### Pitfall 5: Breaking Docs Generation With A Clever New Compare Row Format
**What goes wrong:** The compare snapshot refresh succeeds, but `docs/benchmarks.md` silently drops
rows or renders incomplete output.

**Why it happens:** `tools/docsgen/docsgen.cpp` parses benchmark rows with a strict regex for the
current `emel.cpp ... llama.cpp ... ratio=...x` format.

**How to avoid:** Either preserve that row shape or update docsgen and its expectations in the same
phase.

**Warning signs:** `scripts/generate_docs.sh --check` reports out-of-date docs or the generation
rows disappear from `docs/benchmarks.md`.

### Pitfall 6: Treating Warning-Only Quality Gates As Phase Completion Proof
**What goes wrong:** The repo-wide gate exits successfully with a benchmark warning, and the phase
is incorrectly treated as verified.

**Why it happens:** `scripts/quality_gates.sh` currently soft-fails benchmark drift and only emits
`warning: benchmark snapshot regression ignored by quality gates`.

**How to avoid:** Keep Phase 13’s acceptance criteria tied to explicit benchmark artifact checks,
not to the wrapper’s current warning policy.

**Warning signs:** The only evidence for BENCH-01/02/03 is a green `quality_gates.sh` exit code.

## Code Examples

Verified patterns from repo sources:

### Canonical Compare Row Source

```cpp
// Source: tools/bench/bench_main.cpp
std::printf("%s emel.cpp %.3f ns/op, llama.cpp %.3f ns/op, ratio=%.3fx\n",
            emel_entry.name.c_str(),
            emel_entry.ns_per_op,
            ref_entry.ns_per_op,
            ratio);
```

### Canonical Generation Cases

```cpp
// Source: tools/bench/bench_cases.hpp
inline constexpr std::string_view k_generation_case_name =
  "generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1";
inline constexpr std::string_view k_generation_long_case_name =
  "generation/preloaded_request/llama_68m_prompt_hello_max_tokens_8";
```

### Existing Flash Dispatch Truth Signal

```cpp
// Source: src/emel/generator/sm.hpp
uint64_t generation_flash_attention_dispatch_calls() const noexcept {
  return this->context_.compute.backend.flash_attention_dispatch_calls;
}
```

### Existing Compare Docs Parser Constraint

```cpp
// Source: tools/docsgen/docsgen.cpp
std::regex line_re(
    R"(^([^ ]+) emel\.cpp ([0-9.]+) ns/op, llama\.cpp ([0-9.]+) ns/op, ratio=([0-9.]+)x$)");
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Generic non-flash canonical generation benchmark row only | Current runtime path now uses shipped flash attention, but artifacts still publish generic generation rows | Flash milestone Phases 10-12.1 completed on 2026-03-22 | Phase 13 must make the published evidence catch up to the shipped runtime truth. |
| Benchmark truth depended on benchmark-specific seam audits only | Paritychecker now publishes normal-surface flash proof with `flash_dispatch_calls` and reference identity | Phase 12 completed on 2026-03-21 | Phase 13 has a repo-local pattern to follow for truthful flash evidence without widening runtime scope. |
| Benchmark docs only needed current compare rows | BENCH-02 now requires separating flash evidence from pre-flash baseline | Phase 13 roadmap requirement | Artifact design matters; overwriting `benchmarks_compare.txt` is no longer sufficient by itself. |
| Local `tmp/llama.cpp` fallback was tolerable | Published benchmark evidence now needs deterministic reference sourcing | Phase 13 planning context | The current `tools/bench` local override is now a truthfulness risk, not just a convenience. |

**Deprecated/outdated:**

- Keeping the old baseline only in git history is now outdated for BENCH-02 and BENCH-03.
- Treating the optional bench seam audit as sufficient publication proof is outdated once the
  milestone explicitly promises maintained flash evidence.
- The sample row in `docs/benchmarking.md` is stale relative to current snapshot numbers and should
  not be treated as benchmark evidence.

## Open Questions

1. **Should benchmark reference policy stay pinned or align to paritychecker’s latest-upstream flow?**
   - What we know: `tools/bench/reference_ref.txt` pins benchmarks to
     `ecbcb7ea9d3303097519723b264a8b5f1e977028`, and `docs/benchmarking.md` describes that as the
     reproducible benchmark contract.
   - What's unclear: Whether Phase 13 should keep that reproducibility policy or align benchmark
     publication to Phase 12’s latest-upstream parity policy.
   - Recommendation: Keep benchmarks pinned for reproducibility, but remove the `tmp/llama.cpp`
     local override so the pin is actually authoritative.

2. **What artifact shape should distinguish flash evidence from non-flash baseline?**
   - What we know: Current docsgen only renders one compare table from one snapshot file, and the
     old non-flash short-case baseline currently lives only in git history.
   - What's unclear: Whether the best shape is flash-labeled case names, a sidecar baseline
     snapshot, or multiple generated benchmark tables.
   - Recommendation: Decide this in Wave 0. My recommendation is a maintained sidecar baseline
     artifact plus generated docs sections, because it preserves the canonical row shape while
     keeping history explicit.

3. **Which maintained case should BENCH-03 gate on?**
   - What we know: The short `max_tokens=1` case has a durable pre-flash baseline in git history.
     The long `max_tokens=8` case appears only in the current flash-era snapshot.
   - What's unclear: Whether the long case should be part of the mandatory improvement claim or
     remain supplementary evidence.
   - Recommendation: Gate BENCH-03 on the short case and treat the long case as secondary evidence
     unless an explicit non-flash baseline is reconstructed and maintained for it.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | doctest for repo tests plus shell/CLI workflow assertions |
| Config file | none for bench workflow; repo test entrypoints are script-driven |
| Quick run command | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare` |
| Full suite command | `scripts/quality_gates.sh` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BENCH-01 | Canonical compare workflow runs the shipped flash path on the maintained Llama-68M workload. | integration | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare | rg '^generation/preloaded_request/.* ratio='` plus a flash-proof assertion command in the same workflow | ❌ Wave 0 |
| BENCH-02 | Maintained snapshots/docs distinguish current flash evidence from prior non-flash baseline. | integration | `scripts/bench.sh --compare-update && scripts/generate_docs.sh && rg 'flash|baseline' snapshots/bench docs/benchmarks.md` | ❌ Wave 0 |
| BENCH-03 | At least one maintained canonical case proves flash is faster than the pre-flash baseline. | integration | a repo-local comparator command that reads the preserved non-flash baseline artifact and current flash snapshot and fails unless `flash_ns < baseline_ns` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** run the low-iteration compare command and the targeted artifact/proof check
  for the area being changed.
- **Per wave merge:** run the compare command, docs generation, and the new baseline-comparison
  command together.
- **Phase gate:** full `scripts/quality_gates.sh` plus explicit Phase 13 artifact checks must pass
  before `/gsd:verify-work`.

### Wave 0 Gaps

- [ ] Benchmark compare workflow lacks a durable publish-time assertion that the canonical EMEL
  generation case executed flash attention.
- [ ] Maintained artifacts do not currently preserve the pre-flash short-case baseline separately
  from the current flash-era compare snapshot.
- [ ] `tools/docsgen/docsgen.cpp` only knows how to render one compare table from one strict row
  shape; BENCH-02 likely needs docsgen work.
- [ ] `tools/bench/CMakeLists.txt` still prefers local `tmp/llama.cpp`, which blocks deterministic
  benchmark publication and already caused a local configure failure during research.
- [ ] No existing automated check compares the current canonical flash benchmark against a
  preserved non-flash baseline artifact for BENCH-03.

## Sources

### Primary (HIGH confidence)
- `.planning/STATE.md` - current phase focus, milestone decisions, and blockers.
- `.planning/ROADMAP.md` - Phase 13 goal and success criteria.
- `.planning/REQUIREMENTS.md` - BENCH-01, BENCH-02, and BENCH-03 definitions.
- `.planning/PROJECT.md` - current milestone status and Phase 13 benchmark-evidence intent.
- `tools/bench/bench_main.cpp` - compare row contract and canonical generation-row presence checks.
- `tools/bench/bench_cases.hpp` - canonical generation case names.
- `tools/bench/generation_bench.cpp` - benchmark workload contract, seam audit, and EMEL/reference
  case wiring.
- `tools/bench/CMakeLists.txt` - current reference-source selection and local `tmp/llama.cpp`
  preference.
- `scripts/bench.sh` - authoritative compare/snapshot/update workflow.
- `snapshots/bench/benchmarks.txt` and `snapshots/bench/benchmarks_compare.txt` - current durable
  artifacts.
- `tools/docsgen/docsgen.cpp` and `docs/templates/benchmarks.md.j2` - generated benchmark-doc
  pipeline and row-format parser constraints.
- `docs/benchmarking.md` and `docs/benchmarks.md` - current operator docs and published evidence.
- `src/emel/generator/detail.hpp` and `src/emel/generator/sm.hpp` - shipped flash dispatch counter
  and runtime path.
- `tools/paritychecker/parity_runner.cpp` and `tools/paritychecker/paritychecker_tests.cpp` -
  existing normal-surface flash-proof publication reference.
- Git history: `git show 2acd4fe^:snapshots/bench/benchmarks_compare.txt` and
  `git show 2acd4fe^:snapshots/bench/benchmarks.txt` - durable pre-flash short-case baseline.

### Secondary (MEDIUM confidence)
- `.planning/phases/07-generation-benchmark-harness/07-RESEARCH.md` - original canonical benchmark
  workload contract.
- `.planning/phases/08-generation-compare-output/08-RESEARCH.md` and
  `.planning/phases/08-generation-compare-output/08-VERIFICATION.md` - compare-row publication
  contract and stderr-only seam-audit precedent.
- `.planning/phases/09-benchmark-integration-hardening/09-RESEARCH.md` - current snapshot/docs
  integration model.
- `.planning/phases/12-parity-and-verification-closure/12-RESEARCH.md` and
  `.planning/phases/12-parity-and-verification-closure/12-02-SUMMARY.md` - parity truthfulness
  pattern and note that `tools/bench` still configures from `tmp/llama.cpp`.

### Tertiary (LOW confidence)
- Live local command attempt on 2026-03-22:
  `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`
  - useful as evidence that the current local workflow can fail through `tmp/llama.cpp`, but the
    failure mode may vary by machine.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - direct repo inspection of the active benchmark, snapshot, and docs
  pipeline.
- Architecture: MEDIUM - the truthful workflow constraints are clear, but the exact BENCH-02
  artifact shape is still a planner choice.
- Pitfalls: HIGH - direct code inspection, git-history comparison, and a live local bench failure
  exposed the main risks.

**Research date:** 2026-03-22
**Valid until:** 2026-04-05
