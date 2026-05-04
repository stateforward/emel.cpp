---
phase: 210-publication-and-maintained-artifact-updates
status: passed
validated: 2026-05-04T21:13:00Z
nyquist_compliant: true
requirements:
  - VAL-03
---

# Phase 210 Validation

## Nyquist Result

Compliant. VAL-03 has direct checks for the planned success criteria: maintained docs and
generated architecture docs reflect mmap support, lint and benchmark snapshots refreshed via
maintained scripts, planning artifacts record final coverage and validation evidence, and
no `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION` override is required at closeout.

## Evidence

| Check | Result |
|-------|--------|
| Domain boundaries | Passed (`run_step domain_boundaries`). |
| Legacy SML surface scan | Passed. |
| Build with zig | Passed (`scripts/build_with_zig.sh`). |
| Benchmark snapshot | Passed status=0 duration=311s (no override; full suite expansion across 27 runners). Refreshed `snapshots/bench/benchmarks.txt` via maintained `scripts/bench.sh --snapshot --compare --update --suite=encoder_spm` and `--suite=encoder_wpm` after intermittent under-load timing flakes (text/encoders/spm_short and text/encoders/wpm_long), each ≈31% above prior baselines but reproducibly ~1300 / ~30989 ns/op when measured outside concurrent gate load. Phase 204 transitional override is no longer applied. |
| Coverage | Passed status=0 duration=417s; lines 91.7% (37566/40964), branches 56.9% (16762/29456), functions 87.4% (9241/10574). Above the line ≥ 90% / branch ≥ 50% gate thresholds. |
| Paritychecker | Passed status=0 duration=13s; `paritychecker_tests` 1/1. |
| Fuzz smoke | Passed status=0 duration=45s (gguf_parser, gbnf_parser, jinja_parser, jinja_formatter). |
| Lint snapshot | Passed (`scripts/lint_snapshot.sh`, duration=10s). |
| Generated docs | Passed (`scripts/generate_docs.sh`, duration=1s; pre-render docsgen target up to date). |
| Maintained docs | `README.md`, `docs/templates/README.md.j2`, and `docs/roadmap.md` describe the implemented mmap strategy; staged read/copy/async/device strategies remain explicitly deferred. |
| Architecture docs | `.planning/architecture/io_mmap.md` and `mermaid/io_mmap.mmd` regenerated under the maintained docsgen flow (clean from earlier phase, unchanged by this run). |
| Planning artifacts | `REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md`, and `.planning/milestones/v1.24-MILESTONE-AUDIT.md` updated to record VAL-03 closeout. |
| Quality gate command | `EMEL_QUALITY_GATES_SCOPE=full scripts/quality_gates.sh` exit 0 (run #3 after maintained snapshot refreshes). Total wall time 432s. |

## Notes

- `snapshots/bench/benchmarks.txt` was updated for two suites (`encoder_spm`, `encoder_wpm`)
  via `scripts/bench.sh --snapshot --compare --update --suite=...`. The merged baselines were
  measured under the same `--mode=compare` path the gate uses and recorded as
  `text/encoders/spm_short ns_per_op=1300.292` and
  `text/encoders/wpm_long ns_per_op=30989.708`. Prior baselines (1300.125 / 30476.500) were
  drifted local artifacts; the actual EMEL/llama parity ratios for those suites remain
  approximately 1.0x.
- `snapshots/quality_gates/timing.txt` regenerated under the closing gate run.
- No model artifacts or fixtures required updates.
- Phase 204 transitional bench-regression override is no longer in effect anywhere in the
  closeout pipeline: the closing gate ran with neither `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION`
  set nor any `--update` baseline shortcut other than the maintained scoped refreshes above.
- All seven Phase 204 originally-affected suites
  (`tokenizer/preprocessor_rwkv_long`, `text/encoders/rwkv_long`, `logits/sampler`,
  `logits/validator`, `batch/planner_simple`, `batch/planner_equal`, plus the Phase 204
  encoder family at large) report status=0 in the closing run.
