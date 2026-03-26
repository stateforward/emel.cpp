# Requirements: EMEL

**Defined:** 2026-03-25
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.

## v1 Requirements

### Quantized Path Audit

- [x] **AUD-01**: The canonical Llama-68M ARM generation slice has a maintained operand-path audit
  that classifies each quantized stage as native quantized, approved dense-f32-by-contract, or
  disallowed fallback.

### Path Closure

- [x] **PATH-01**: Supported canonical ARM quantized requests do not silently widen through
  disallowed whole-row or operator-level dequantize-to-f32 substitution in the shipped runtime
  path.
- [x] **PATH-02**: Unsupported or not-yet-ported quantized cases publish explicit no-claim
  behavior instead of silently routing through a misleading f32 fallback path.

### Runtime Attribution

- [x] **ATTR-01**: Maintained paritychecker and benchmark outputs publish enough attribution to
  prove whether the canonical ARM request stayed on the approved quantized path.

### Verification

- [x] **VER-04**: Kernel, runtime, and regression tests cover the audited quantized-path branches
  and fail if supported canonical requests regress back to disallowed f32 fallback.
- [x] **PAR-05**: `tools/paritychecker --generation` proves the canonical ARM workload stays on the
  maintained approved quantized path across `1`, `10`, `100`, and `1000` tokens.

### Benchmarking

- [x] **BENCH-10**: Maintained benchmark evidence isolates the end-to-end impact of full
  quantized-path closure from remaining generator-side ARM math cost.

## v2 Requirements

### Generator Math

- **GEN-03**: Optimize ARM generator-side RMSNorm, RoPE, residual-add, and SwiGLU math after the
  quantized path contract is fully proven and measured.

### Broader Coverage

- **FLASH-03**: Broaden flash attention beyond the canonical Llama-68M shape and workload
  contract.
- **MODEL-01**: Roll optimized ARM flash attention out to additional model fixtures after the
  canonical path remains correct and benchmarked.

### Benchmark Policy

- **BENCH-07**: Revisit whether noisy benchmark drift should become a blocking repo gate once ARM
  compare evidence stabilizes.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Broad generator-side ARM math optimization beyond proving or closing the quantized-path contract | Keep this milestone fixed on operand-path fidelity before chasing the next hotspot. |
| Broader model matrix or non-canonical workload claims | The canonical ARM Llama-68M slice remains the maintained truth anchor for this milestone. |
| Non-ARM backend quantized specialization | The immediate question is whether the shipped ARM path still hides disallowed f32 fallback. |
| Whole-program orchestration rewrites | The audit targets data-plane fidelity, not actor-model redesign. |
| Benchmark gate hardening | Drift policy should stay stable until maintained quantized-path evidence is settled. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| AUD-01 | Phase 22 | Complete |
| PATH-01 | Phase 23 | Complete |
| PATH-02 | Phase 22 | Complete |
| ATTR-01 | Phase 24 | Complete |
| VER-04 | Phase 24 | Complete |
| PAR-05 | Phase 24 | Complete |
| BENCH-10 | Phase 25 | Complete |

**Coverage:**
- v1 requirements: 7 total
- Mapped to phases: 7
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-25*
*Last updated: 2026-03-25 after completing Phase 25*
