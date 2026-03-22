# Phase 12: Parity And Verification Closure - Research

**Researched:** 2026-03-21
**Domain:** paritychecker flash-proof publication and latest-upstream parity validation for the
canonical Llama-68M slice
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Phase Boundary

Phase 12 proves that the shipped flash-attention path is truthfully exercised on the maintained
`tools/paritychecker --generation` surface for the canonical Llama-68M slice and remains parity
stable against the latest upstream `llama.cpp` selected through the paritychecker CMake flow. This
phase stays inside parity and test closure: it does not broaden public/runtime surface area, and it
does not publish benchmark evidence yet.

## Implementation Decisions

### Reference Alignment Target
- Phase 12 parity must build the reference implementation through CMake against the latest upstream
  `llama.cpp` state instead of reading from a local `tmp/llama.cpp` checkout.
- Upstream drift blocks Phase 12 completion.
- Local `tmp/llama.cpp` state must not decide parity results.
- Parity success still means exact token/output parity.

### Flash Proof Surface
- The normal `paritychecker --generation` success surface must expose flash-execution proof
  directly.
- A clearer dedicated proof block is allowed.
- The surface must expose at least flash-dispatch counts.
- Failures should auto-emit enough diagnostics to explain the failure without a second rerun.

### Verification Contract
- Gate on both the short canonical run and one bounded longer decode.
- Both workloads must pass.
- Keep the paritychecker surface success-path focused; do not explode into a broad failure matrix.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PAR-01 | `tools/paritychecker --generation` can prove that the flash-attention path executed on `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`. | Phase 11 already added durable `generation_flash_attention_dispatch_calls()` observability. `tools/paritychecker/parity_runner.cpp` already reads it and can publish it on the normal surface, not only in `--dump`. |
| PAR-02 | The canonical flash-attention generation result remains parity-stable against an aligned `llama.cpp` reference configuration. | `run_generation_harness_contract(...)` already compares EMEL and reference output for bounded generation. The remaining work is to ensure the reference comes from the CMake-fetched latest upstream path and to extend coverage from the one-token smoke case to the bounded longer decode. |
| VER-01 | Tests cover shared-kernel correctness, generator adoption, and negative selection or fallback behavior for the flash-attention path. | Phase 10 and 11 already cover kernel correctness, generator adoption, and deterministic negative selection. Phase 12 mainly needs paritychecker subprocess coverage for the latest-reference proof surface and longer decode. |
</phase_requirements>

## Summary

The existing paritychecker generation contract was already close to the Phase 12 goal. It loaded
the canonical fixture, required successful EMEL generation, required
`generation_flash_attention_dispatch_calls() > 0`, and compared bounded EMEL output with a direct
`llama.cpp` reference run. The main gaps were around truthfulness and contract shape:

1. `tools/paritychecker/CMakeLists.txt` still preferred a repo-local `tmp/llama.cpp` checkout,
   which made the parity surface machine-dependent instead of truthfully using a fetched latest
   upstream reference.
2. The richer proof block (`reference_decode_seams`, `kernel_dispatch`, `flash_dispatch`) only
   appeared in `--dump` mode, while the normal success path exposed only the one-line result.
3. Automated subprocess coverage only locked the short one-token generation request, not the
   bounded longer decode already used elsewhere in the milestone.

The narrowest truthful implementation is therefore:
- remove the local `tmp/llama.cpp` preference from paritychecker CMake
- always fetch the reference through CMake with `master` as the default ref
- expose the actual fetched reference commit on the normal proof surface
- print the proof block on successful generation and on post-initialize failures
- extend `paritychecker_tests` to require the bounded longer decode contract

**Primary recommendation:** keep the existing `run_generation_harness_contract(...)` as the single
truth source, but harden it so the normal paritychecker surface always tells the truth about which
reference it built against and whether flash actually executed.

## Project Constraints (from AGENTS.md)

- Keep the milestone inside the existing paritychecker tool surface.
- Do not introduce a broader runtime or API boundary.
- Preserve truthful flash claims and deterministic failure reporting.
- Run `scripts/quality_gates.sh` after implementation changes.
- Avoid unrelated snapshot/doc churn unless directly required by the phase.

## Standard Stack

### Core
| Library / Component | Version | Purpose | Why Standard |
|---------------------|---------|---------|--------------|
| `tools/paritychecker/CMakeLists.txt` | repo local | Reference source selection | This is where parity decides whether it builds against local tmp state or a fetched upstream checkout |
| `tools/paritychecker/parity_runner.cpp` | repo local | Generation parity contract and proof surface | This is already the single bounded-generation truth source |
| `tools/paritychecker/paritychecker_tests.cpp` | repo local | Subprocess contract assertions | This already checks normal and dump generation output |

### Supporting
| Library / Component | Version | Purpose | When to Use |
|---------------------|---------|---------|-------------|
| `src/emel/generator/sm.hpp` | repo local | Flash-dispatch proof counter | Use as the authoritative EMEL-side proof of flash execution |
| `tools/paritychecker/reference_ref.txt` | repo local | Historical parity pin artifact | Still present in the repo, but the user-directed Phase 12 contract is that parity fetches latest upstream through CMake rather than preferring this file or local tmp state |
| `scripts/quality_gates.sh` | repo local | Full regression gate | Required after implementation |

## Architecture Patterns

### Pattern 1: Keep One Generation Truth Surface
Use `run_generation_harness_contract(...)` as the single parity contract rather than creating a
second proof-only path. This keeps Phase 12 truthful and small.

### Pattern 2: Publish Proof By Reusing Existing Seam Dumps
`dump_reference_decode_seam(...)` already prints the useful flash and reference seam counters. The
best Phase 12 surface is to promote that block into the normal success/failure contract instead of
inventing new diagnostics from scratch.

### Pattern 3: Latest Reference Must Be Build-Time Truth
If parity is supposed to track the latest upstream `llama.cpp`, that choice has to live in
`tools/paritychecker/CMakeLists.txt`, not in user-local workspace state.

### Anti-Patterns To Avoid
- Preferring `tmp/llama.cpp` when present.
- Hiding flash proof behind `--dump` while claiming normal parity success.
- Deferring the bounded longer decode even though the phase explicitly gates on it.
- Expanding into benchmark publication or benchmark artifact churn during Phase 12.

## Validation Architecture

- **Quick feedback target:** `ctest --test-dir build/paritychecker_zig_latest --output-on-failure -R paritychecker_tests`
- **Direct proof command:** `./build/paritychecker_zig_latest/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 8`
- **Full regression target:** `scripts/quality_gates.sh`
- **Primary proof points:**
  - normal generation output includes flash proof and reference identity
  - bounded long decode stays parity-stable against the fetched upstream reference
  - post-initialize failures auto-emit enough proof/dump surface to explain what broke

## Suggested Plan Shape

Two plans are enough:

1. **Latest-upstream parity source and proof publication**
   - remove `tmp/llama.cpp` preference from paritychecker CMake
   - publish reference identity and seam proof on the normal generation surface
   - auto-emit useful diagnostics on post-initialize failures

2. **Bounded parity coverage and regression verification**
   - extend paritychecker subprocess tests to lock the longer bounded decode
   - keep the normal and dump proof surfaces covered
   - run paritychecker tests plus full repo quality gates

---
*Phase: 12-parity-and-verification-closure*
*Researched: 2026-03-21*
