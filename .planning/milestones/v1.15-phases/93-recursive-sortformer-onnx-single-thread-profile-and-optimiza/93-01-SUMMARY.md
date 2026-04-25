---
phase: 93
plan: 1
status: complete
completed: 2026-04-24
requirements-completed:
  - PRF-01
  - PRF-02
  - BEN-01
  - DOC-01
one-liner: EMEL beats ONNX CPU single-thread while exact-matching PyTorch/NeMo and ONNX.
---

# Phase 93 Summary

Phase `93` is complete. The retained optimization pre-normalizes attention scores once per head,
removing the repeated division from the weighted value accumulation while keeping the existing
runtime-owned Sortformer encoder path.

The ONNX reference runner now records the actual provider list returned by ONNX Runtime. The final
strict generated evidence reports `actual_providers=CPUExecutionProvider` with
`thread_contract=intra_op=1 inter_op=1 execution_mode=sequential`.

Final strict generated record directory:

`build/diarization_compare_phase93_score_prenorm_restored_pytorch`

Final timing and parity:

- EMEL: `1352780166 ns/op`, `15` runs, `output_dim=17`, checksum `4249677247906920305`
- ONNX benchmark reference: `1920646958 ns/op`, `15` runs, `output_dim=17`,
  checksum `4249677247906920305`
- PyTorch/NeMo parity reference: `9420665125 ns/op`, `1` run, `output_dim=17`,
  checksum `4249677247906920305`

EMEL is `0.704x` ONNX time on the strict record set, so the recursive optimization stop condition
is satisfied.

Rejected candidates:

- A 4-wide attention weighted-value rewrite preserved parity but regressed EMEL runtime and was
  reverted.
- A kernel-owned NEON weighted-value helper preserved parity but regressed EMEL runtime and was
  reverted.
- Increasing the prepared F32 LHS kernel depth block to `512` did not hold up under the strict
  PyTorch/ONNX lane and was reverted to the retained `256` block.
