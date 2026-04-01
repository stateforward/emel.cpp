# Requirements: EMEL

**Defined:** 2026-03-31
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.

## v1 Requirements

### Initializer Orchestration

- [ ] **INIT-01**: Generator owns an explicit `src/emel/generator/initializer` orchestration
      machine for conditioner binding, renderer initialization, memory reserve, graph reserve, and
      sampling configuration.
- [ ] **INIT-02**: Initialize request-scoped orchestration data stays on typed runtime/internal
      events instead of being mirrored into generator context phase fields or flags.

### Explicit Modeling

- [ ] **INIT-03**: Initialize route selection remains explicitly modeled through guards, states,
      and transitions, with no hidden runtime branching added to generator actions or detail
      helpers.

### Maintainability And Proof

- [ ] **ARCH-02**: The top-level generator initialize/publication surface is materially smaller
      and easier to inspect after the initializer extraction.
- [ ] **VERIFY-02**: Maintained generator, paritychecker, benchmark, and quality-gate coverage
      stay green on the current Llama and canonical Qwen slices across the initializer boundary.

## v2 Requirements

### Further Generator Decomposition

- **DECODE-01**: Revisit generator decode decomposition only after v1.8 proves a cold-path
  submachine without reintroducing the earlier per-token performance regression.
- **ATTN-01**: Revisit attention-family decomposition such as `flash` vs `nonflash` through
  `sm_any` only after the remaining generator routing shape is clearer.

### Product Scope

- **MODEL-02**: Broaden beyond the canonical Qwen3-0.6B fixture to additional Qwen architectures
  or quantizations once the maintained generator decomposition work is proven.
- **COND-02**: Add richer Qwen chat or tool-calling request surfaces only after the canonical
  maintained slice has an explicit and stable conditioning contract.

### Performance And Policy

- **GEN-03**: Optimize remaining generator-side hot spots after the maintained decomposition work
  is correct, parity-backed, and benchmarked.
- **BENCH-07**: Revisit whether noisy benchmark drift should become a blocking repo gate once the
  maintained compare surfaces are stable enough to justify it.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Decode submachine extraction in v1.8 | Prior decode decomposition work regressed hot-path performance; keep the per-token decode loop on the parent generator in this milestone. |
| Internal request-flow or `preprocessor` machine split | The user chose a single-machine milestone focused on `generator/initializer`. |
| Separate session/runtime actor redesign | This milestone decomposes `generator`, not ownership around it. |
| Attention-family `sm_any` extraction in v1.8 | Keep the slice narrow to initializer extraction and proof. |
| Broader Qwen-family or richer request-surface work | Keep the acceptance boundary on the current maintained Llama and canonical Qwen anchors. |
| Benchmark gate hardening | Existing warning-only benchmark debt is separate from this generator refactor. |
| Hidden control-flow shortcuts in actions/detail helpers | The milestone must preserve explicit behavior modeling rather than trading structure for convenience. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INIT-01 | Phase 33 | Pending |
| INIT-02 | Phase 33 | Pending |
| INIT-03 | Phase 33 | Pending |
| ARCH-02 | Phase 34 | Pending |
| VERIFY-02 | Phase 34 | Pending |

**Coverage:**
- v1 requirements: 5 total
- Mapped to phases: 5
- Unmapped: 0

---
*Requirements defined: 2026-03-31*
*Last updated: 2026-03-31 after defining v1.8 initializer scope*
