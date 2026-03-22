# Deferred Items

## 2026-03-09

- `scripts/quality_gates.sh` fails in the coverage-report step after `emel_tests` passes because
  `gcovr` attempts to `chdir` into a missing
  `build/coverage/CMakeFiles/emel_tests_bin.dir/tests/kernel` directory. This is outside plan
  `07-01`, which only adds the generation benchmark harness in `tools/bench/`.
