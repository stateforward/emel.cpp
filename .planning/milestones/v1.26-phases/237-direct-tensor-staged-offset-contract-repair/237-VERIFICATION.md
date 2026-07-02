# Phase 237 Verification

status: passed

All commands were run from:
`/Users/gabrielwillen/.atmux/teams/emel_cpp/milestone63/worktree`

## Source Repair

Phase 237 closes the audit gap where direct
`model/tensor::event::request_staged_load` forwarded the whole-file
`source_buffer` to `io/staged_read` while also carrying a nonzero
`file_offset`. The repaired direct path now validates that the supplied source
buffer covers `file_offset + byte_size`, passes `source_buffer + file_offset` as
the staged copy `source_span`, and sets `source_span_bytes` to the logical
window length.

## Reproduction

The regression test was added before the implementation change:

```bash
./build/emel_tests_bin --test-case="model_tensor_request_staged_load_applies_nonzero_file_offset"
```

Result before repair: **FAIL** (exit `1`).

Observed failure: `machine.process_event(request)` returned false, the `_error`
callback fired, no bytes were copied, and the tensor remained unbound.

## Passing Evidence

```bash
cmake --build build --target emel_tests_bin
./build/emel_tests_bin --test-case="model_tensor_request_staged_load_applies_nonzero_file_offset"
./build/emel_tests_bin --test-case="model_tensor_request_staged_load_*"
ctest --test-dir build -R '^emel_tests_model_and_batch$' --output-on-failure
```

Results:

- `cmake --build build --target emel_tests_bin`: **PASS** (exit `0`)
- Offset regression doctest: **PASS** (`1/1` test, `12/12` assertions)
- Staged tensor doctest subset: **PASS** (`4/4` tests, `37/37` assertions)
- `emel_tests_model_and_batch` CTest shard: **PASS** (`1/1` tests)

## Quality Gate

```bash
EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/model/tensor/actions.hpp src/emel/model/tensor/guards.hpp tests/model/tensor/lifecycle_tests.cpp .planning/phases/237-direct-tensor-staged-offset-contract-repair/237-CONTEXT.md .planning/phases/237-direct-tensor-staged-offset-contract-repair/237-01-PLAN.md" scripts/quality_gates.sh
```

Result: **PASS** (exit `0`).

Relevant lane summaries:

- Legacy SML surface scan passed.
- `bench_snapshot`: `status=0`
- `test_with_coverage`: `status=0`
- `paritychecker`: `status=0`
- `fuzz_smoke`: skipped because there were no fuzz-affecting changed files.
- `generate_docs`: skipped because there were no docsgen-affecting changed files.

## Requirement Evidence

- `TNX-01`: direct tensor staged-load still dispatches through injected public
  `emel::io::staged_read::sm` with `process_event(...)`; no private cross-actor
  calls were introduced.
- `TNX-03`: success is observable through `request_staged_load_done`, tensor
  `resident` state inspection, and copied offset-window bytes.
- `TNX-04`: staged validation failure remains observable through
  `request_staged_load_error` and explicit tensor error mapping.
- `TST-01`: public dispatch success doctest covers nonzero-offset staged load.
- `TST-02`: public dispatch failure doctest covers staged contract rejection via
  the direct tensor route.
