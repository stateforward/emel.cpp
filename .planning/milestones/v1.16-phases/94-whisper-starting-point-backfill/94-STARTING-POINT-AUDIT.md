# Phase 94 Starting-Point Audit

Date: 2026-04-25
Scope: local Whisper-started files and adjacent benchmark/parity artifacts referenced by
`.planning/ROADMAP.md` Phase 94 success criteria.

## Classification Ledger

| File | Classification | Source-backed state | Phase 94 disposition |
|------|----------------|---------------------|----------------------|
| `CMakeLists.txt` | landed | `src/emel/model/whisper/detail.cpp` is now compiled into `emel`, so Whisper loader/contract code is actually built. | Keep as baseline. |
| `src/emel/model/architecture/detail.cpp` | landed | Registers `"whisper"` in `default_architecture_span()` with Whisper `load_hparams` and `validate_execution_contract` hooks. | Keep as baseline. |
| `src/emel/model/whisper/detail.hpp` | landed | Declares Whisper execution contract and validation surface with explicit constants and tensor-family views. | Keep as baseline. |
| `src/emel/model/whisper/detail.cpp` | landed | Enforces tiny-contract checks (`n_mels=80`, `n_vocab=51865`, canonical tensor shapes/prefix families). | Keep as baseline. |
| `tests/model/loader/lifecycle_tests.cpp` | landed | Adds focused loader/contract acceptance/rejection tests for Whisper tiny metadata and tensor contract. | Keep as baseline. |
| `src/emel/model/data.hpp` | keep-and-fix | Adds `is_whisper_execution_architecture(...)` API, but broader model execution classification remains llama-centric naming and needs cleanup when Whisper runtime lands. | Keep now; refactor during kernel/runtime phases. |
| `src/emel/model/data.cpp` | keep-and-fix | Adds Whisper architecture helpers while still carrying cross-architecture quantized-path audit logic not yet tied to Whisper runtime orchestration. | Keep now; tighten architecture-specific routing in later phases. |
| `src/emel/kernel/detail.hpp` | keep-and-fix | Adds/extends native quant dtype and dequant helpers (`q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, `q8_k`) but remains partial until wired through Whisper runtime path ownership/tests. | Keep now; complete in Phase 96. |
| `tests/models/README.md` | keep-and-fix | Whisper fixture provenance is pinned, but wording needs explicit variant-family and loader-only scope guardrails. | Corrected in Phase 94. |
| `tests/model/fixture_manifest_tests.cpp` | keep-and-fix | Verifies fixture facts, but did not yet lock wording that prevents parity/runtime overclaims. | Corrected in Phase 94. |
| `scripts/bench_whisper_reference_whisper_cpp.sh` | keep-and-fix | Useful isolated reference-lane wrapper; currently a standalone helper not yet part of the maintained parity+benchmark record pipeline. | Keep now; integrate under parity/benchmark phases. |
| `tools/paritychecker/parity_runner.cpp` | replace | Current implementation is generation-oriented (`generation_fixture_registry`, tokenizer/logits flow) and not a dedicated Whisper ASR parity comparator. | Replace with Whisper ASR parity lane in Phase 99. |
| `.tmp-phase92-sortformer_probe` / `.tmp-phase92-sortformer_probe.cpp` / `a.out` | discard | Scratch/probe artifacts, not maintained Whisper milestone evidence. | Exclude from milestone claims. |

## Wording Corrections Required By Phase 94

1. Replace q80-only framing with explicit **variant-family** scope language where fixture docs are used
   as milestone truth.
2. State explicitly that current Whisper loader/contract evidence is **not** a claim that EMEL ASR
   runtime or `whisper.cpp` parity is already complete.
3. Add fixture-manifest assertions so those scope statements remain protected by CI.

## Kernel Compliance Review (BACK-03)

Reviewed file: `src/emel/kernel/detail.hpp`

- No queue/defer usage and no actor `process_event(...)` calls were introduced in kernel helpers.
- Added quant helpers are data-plane compute/dequant paths (bounded loops and dtype arithmetic), not
  SML orchestration code.
- This preserves the "kernel does numeric work, orchestration decides behavior" split.
- Remaining gap: this phase does **not** yet prove Whisper runtime guard/transition routing over these
  kernels; that proof belongs to Phase 96+ runtime integration and tests.

Conclusion: started kernel edits are acceptable as partial kernel groundwork, but cannot be claimed as
complete maintained Whisper runtime/parity support until explicit SML-routed integration is delivered.
