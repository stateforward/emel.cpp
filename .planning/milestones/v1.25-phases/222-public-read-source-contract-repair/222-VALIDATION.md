---
phase: 222-public-read-source-contract-repair
status: passed
validated: 2026-05-06T04:46:52Z
nyquist_compliant: true
requirements:
  - PLAT-01
  - TIO-03
  - VAL-02
  - VAL-04
---

# Phase 222 Validation

## Nyquist Result

Compliant. Phase 222 closes the source-backed actor-detail reach-through gap by
moving maintained source-byte loading to a public `emel::io::source` setup-time
API and proving maintained lanes no longer include `emel/io/read/detail.hpp`.

## Evidence

| Check | Result |
|-------|--------|
| Public source contract | Passed. `src/emel/io/source/any.hpp` exposes `emel::io::source::load_file_bytes`. |
| Read actor internals | Passed. `src/emel/io/read/detail.hpp` no longer owns the file-byte loading helper. |
| Maintained tools | Passed. Generation, Sortformer, embedded probe, and paritychecker lanes use `emel/io/source/any.hpp`. |
| Guardrails | Passed. Model-loader and paritychecker source guardrails forbid actor read detail reach-through. |
| Focused tests | Passed. Model-and-batch and paritychecker tests pass. |
| Maintained generation flow | Passed. `generation_compare_tests` passes after the reference build cache uses `/usr/bin/git`. |
| Domain boundaries | Passed. `scripts/check_domain_boundaries.sh` exits 0. |
| Quality gate | Passed. Changed-file scoped `scripts/quality_gates.sh` exits 0 without benchmark-regression override. |

## Residual Risk

The maintained reference generation test depends on the local reference build
cache not pointing at the atmux Git shim. Phase 223 should document this in
closeout truth if the cache state affects repeatability.
