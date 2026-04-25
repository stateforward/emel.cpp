---
phase: 93
slug: recursive-sortformer-onnx-single-thread-profile-and-optimiza
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-24
---

# Phase 93 - Validation Strategy

## Completion Preconditions

- [x] `93-01-SUMMARY.md` exists with `requirements-completed` frontmatter.
- [x] `93-VERIFICATION.md` exists and records profile/optimization evidence.
- [x] Roadmap and state no longer list Phase `93` as pending.
- [x] No unresolved manual-only blockers remain in validation scope.

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | yes | The retained optimization is runtime-owned and does not create a separate compute lane. |
| `docs/rules/sml.rules.md` | yes | No queueing, self-dispatch, or hidden runtime-routing changes were introduced. |
| `docs/rules/cpp.rules.md` | yes | Verification includes focused builds, tests, docs, strict compare, and quality gates. |

No rule violations were found within validation scope.

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest, ONNX Runtime, uv-backed PyTorch/NeMo reference, quality gates |
| Quick run command | `build/bench_tools_ninja/diarization_compare_tests` |
| Full suite command | `scripts/quality_gates.sh` |
| Strict compare command | `scripts/bench_diarization_compare.sh --skip-emel-build --setup-pytorch-reference-env --output-dir build/diarization_compare_post_pipeline_pr_feedback --onnx-reference-model build/onnx_ref/diar_streaming_sortformer_4spk-v2.1.onnx --pytorch-reference-model nvidia/diar_streaming_sortformer_4spk-v2.1` |

## Manual-Only Verifications

None. The strict generated compare record proves EMEL `1370917625 ns/op` beats ONNX
`5900446125 ns/op` while exact-matching PyTorch/NeMo and ONNX output.

## Validation Sign-Off

- [x] Completion preconditions satisfied.
- [x] Rule-compliance review recorded.
- [x] Executable verification commands documented.
- [x] No manual-only blockers remain.
- [x] `nyquist_compliant: true` is supported by artifact evidence.

Approval: approved 2026-04-24.
