# Phase 93 Verification

## Source Changes

- `src/emel/diarization/sortformer/encoder/detail.cpp` pre-normalizes attention scores before the
  weighted value accumulation, replacing repeated per-value division with one reciprocal per head.
- `tools/bench/diarization_sortformer_onnx_reference.py` emits the actual ONNX Runtime providers
  returned by `session.get_providers()` in each generated result note.

## Commands

- `cmake --build build/bench_tools_ninja --target bench_runner`
  - Result: passed.
- Reference runner syntax:

  ```bash
  python3 -m py_compile \
    tools/bench/diarization_compare.py \
    tools/bench/diarization_sortformer_onnx_reference.py \
    tools/bench/diarization_sortformer_pytorch_reference.py
  ```

  - Result: passed.
- `cmake --build build/bench_tools_ninja --target bench_runner diarization_compare_tests`
  - Result: passed, no work required.
- `build/bench_tools_ninja/diarization_compare_tests`
  - Result: passed, `7` test cases and `80` assertions.
- `cmake --build build/coverage --target emel_tests_bin`
  - Result: passed.
- `ctest --test-dir build/coverage --output-on-failure -R 'emel_tests_(kernel_and_graph|diarization)'`
  - Result: passed, `2/2` tests.
  - `emel_tests_diarization`: passed in `541.29 sec`.
  - `emel_tests_kernel_and_graph`: passed in `5.08 sec`.
- `scripts/generate_docs.sh`
  - Result: passed.
- `build/docsgen/docsgen --root "$PWD" --check`
  - Result: passed.
- Touched-file whitespace check:

  ```bash
  git diff --check -- \
    tools/bench/diarization_sortformer_onnx_reference.py \
    src/emel/diarization/sortformer/encoder/detail.cpp \
    docs/benchmarking.md \
    docs/benchmarks.md \
    tools/docsgen/docsgen.cpp \
    snapshots/bench/benchmarks.txt \
    snapshots/bench/benchmarks_compare.txt \
    .planning/STATE.md \
    .planning/ROADMAP.md \
    .planning/phases/93-recursive-sortformer-onnx-single-thread-profile-and-optimiza
  ```

  - Result: passed.
- Clean ONNX compare:

  ```bash
  rm -rf build/diarization_compare_phase93_score_prenorm_restored
  EMEL_DIARIZATION_COMPARE_RUNS=15 \
  EMEL_DIARIZATION_COMPARE_WARMUP_RUNS=3 \
  scripts/bench_diarization_compare.sh \
    --skip-emel-build \
    --output-dir build/diarization_compare_phase93_score_prenorm_restored \
    --onnx-reference-model build/onnx_ref/diar_streaming_sortformer_4spk-v2.1.onnx
  ```

  - Result: passed.
  - EMEL: `1351761959 ns/op`, ONNX: `2119327000 ns/op`, exact match.
- Strict PyTorch+ONNX compare:

  ```bash
  rm -rf build/diarization_compare_post_pipeline_pr_feedback
  EMEL_DIARIZATION_COMPARE_RUNS=15 \
  EMEL_DIARIZATION_COMPARE_WARMUP_RUNS=3 \
  scripts/bench_diarization_compare.sh \
    --skip-emel-build \
    --setup-pytorch-reference-env \
    --output-dir build/diarization_compare_post_pipeline_pr_feedback \
    --onnx-reference-model build/onnx_ref/diar_streaming_sortformer_4spk-v2.1.onnx \
    --pytorch-reference-model nvidia/diar_streaming_sortformer_4spk-v2.1
  ```

  - Result: passed.
  - EMEL: `1370917625 ns/op`, ONNX: `5900446125 ns/op`, PyTorch/NeMo: `11417840125 ns/op`.
  - Exact output match against ONNX and PyTorch/NeMo.

## Final Generated Record Facts

From `build/diarization_compare_post_pipeline_pr_feedback`:

| Lane | Role | ns/op | Runs | output_dim | output_checksum | Status |
|------|------|------:|-----:|-----------:|----------------:|--------|
| `emel.diarization.sortformer` | maintained runtime | `1370917625` | `15` | `17` | `4249677247906920305` | `ok` |
| `onnx.sortformer.v2_1` | benchmark reference | `5900446125` | `15` | `17` | `4249677247906920305` | `ok` |
| `pytorch.nemo.sortformer.v2_1` | parity reference | `11417840125` | `1` | `17` | `4249677247906920305` | `ok` |

ONNX note includes:

`proof_status=onnxruntime_cpu thread_contract=intra_op=1 inter_op=1`
`execution_mode=sequential actual_providers=CPUExecutionProvider`

## Acceptance Criteria

1. Reproducible ONNX and EMEL profile command recorded.
   - Passed. Verification records both generated compare commands.
2. Recursive profiling continued until EMEL beat ONNX single-thread.
   - Passed. EMEL is `0.232x` ONNX time in the strict generated record set.
3. Optimizations stayed in kernel/runtime-owned code.
   - Passed. Retained performance change is in the Sortformer encoder runtime path; rejected
     candidate helpers were reverted.
4. Loading/setup cost is separated from steady-state runtime.
   - Passed. Records report `prepare_ns_per_op` separately and compare `ns_per_op` steady-state
     timings.
5. Correctness remained gated by PyTorch/NeMo and ONNX exact match.
   - Passed. Both reference lanes exact-match EMEL output bytes, dimension, and checksum.
