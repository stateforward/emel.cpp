---
phase: 83
status: passed
---

# Phase 83 Verification

## Result

Passed for the split Phase 83 contract-repair scope.

## Passed Checks

| Requirement | Evidence | Status |
|-------------|----------|--------|
| SORT/RUN prerequisite | Self-converted upstream NVIDIA `.nemo` with the OpenResearchTools converter and compared it to the maintained OpenResearchTools GGUF. Size matched at `471107712` bytes; tensor manifest matched; all `1007` tensor payload hashes matched; only `general.name` differed. | passed |
| RUN-02 prerequisite | Maintained profile and tensor-family decisions are now explicit in model/request validation. | passed |
| RUN-03 prerequisite | No benchmark/tool fallback was introduced; contract repair remains in `src/` and tests. | passed |

## Deferred To Decimal Execution Phases

| Requirement | Deferred Work | Phase |
|-------------|---------------|-------|
| RUN-01 | Execute the maintained Conformer encoder from `enc.*` tensors. | 83.1 |
| RUN-01/RUN-02/RUN-03 | Execute `mods.*` projection/head work and speaker cache state. | 83.2 |
| RUN-01/RUN-02/RUN-03 | Execute the maintained `te.*` transformer encoder stack. | 83.3 |
| RUN-01/RUN-02/RUN-03 | Compose the stage-owned execution actor without external fallbacks. | 83.4 |

## Verification Commands

- `uv run --python 3.12 --with torch --with pyyaml --with numpy --with gguf /tmp/emel_sortformer_verify/convert_nemo_sortformer_to_gguf.py --model /tmp/emel_sortformer_verify/diar_streaming_sortformer_4spk-v2.1.nemo --out /tmp/emel_sortformer_verify/converted.gguf --summary-json /tmp/emel_sortformer_verify/converted.summary.json --name diar_streaming_sortformer_4spk-v2.1 --outtype f32`
- Normalized GGUF compare of `/tmp/emel_sortformer_verify/converted.gguf` against
  `/tmp/emel_sortformer_verify/maintained.gguf`
- `cmake --build build/coverage --target emel_tests_bin -j 8`
- `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_(diarization|model_and_batch)$' -j 2`

## Follow-Up

The next phase is 83.1. It must implement the real native Conformer encoder tensor binding and
fixed-profile kernel path under `src/emel/diarization/sortformer/encoder/`. A synthetic projection
or partial head-only execution remains out of scope and must not be landed as RUN-01.
