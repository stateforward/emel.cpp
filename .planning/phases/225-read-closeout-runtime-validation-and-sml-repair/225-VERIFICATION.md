---
phase: 225-read-closeout-runtime-validation-and-sml-repair
verified: 2026-05-06T15:45:45Z
status: passed_with_dyld_fallback
dyld_launch_blocker: true
requirements:
  - VAL-01
  - TIO-03
  - VAL-04
  - VAL-03
---

# Phase 225 Verification

Current source validation was rerun for Plan 06. Direct `build/zig` CTest launch is
environment-sensitive on this macOS host: one focused I/O run passed, while direct
model/batch and combined `build/zig` launches aborted before doctest execution with dyld
shared-cache / `libSystem.B.dylib` output. The same focused shards passed in the
coverage build during the changed-file quality gate.

## Command Results

| Command | Result | Evidence |
|---------|--------|----------|
| `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch` | failed before doctests | dyld launch abort before doctest execution |
| `ctest --test-dir build/zig --output-on-failure -R emel_tests_io` | passed | 1/1 test passed in 0.66 sec |
| `ctest --test-dir build/zig --output-on-failure -R 'emel_tests_(model_and_batch\|io)'` | failed before doctests | both direct launches aborted in dyld |
| `scripts/check_domain_boundaries.sh` | passed | exit 0 |
| `node .codex/get-shit-done/bin/gsd-tools.cjs validate consistency` | passed with warnings | 16 pre-existing roadmap/archive warnings, no errors |
| `EMEL_QUALITY_GATES_CHANGED_FILES='src/emel/io/events.hpp:src/emel/io/read/events.hpp:src/emel/io/read/detail.hpp:src/emel/io/read/guards.hpp:src/emel/io/read/actions.hpp:src/emel/io/read/sm.hpp:src/emel/io/loader/events.hpp:src/emel/io/loader/detail.hpp:src/emel/io/loader/guards.hpp:src/emel/io/loader/actions.hpp:src/emel/io/loader/sm.hpp:src/emel/model/loader/events.hpp:src/emel/model/loader/guards.hpp:src/emel/model/loader/actions.hpp:src/emel/model/loader/sm.hpp:tests/io/read/lifecycle_tests.cpp:tests/io/loader/lifecycle_tests.cpp:tests/model/loader/lifecycle_tests.cpp:tools/bench/generation_bench.cpp:tools/bench/diarization/sortformer_fixture.hpp:tools/embedded_size/emel_probe/main.cpp:tools/paritychecker/parity_engines.cpp' EMEL_QUALITY_GATES_BENCH_SUITE='generation:diarization_sortformer' scripts/quality_gates.sh` | passed | relevant generation and Sortformer benchmark lanes, coverage, paritychecker, docs, and fuzz smoke all passed |

An initial changed-file quality gate without `EMEL_QUALITY_GATES_BENCH_SUITE` also ran
without any benchmark-regression override. It failed only in the unrelated
`text/jinja/formatter_short` benchmark lane after passing focused coverage and
paritychecker. The follow-up command scoped the benchmark suite to Phase 225's maintained
generation and Sortformer tool lanes, as allowed by the repository gate rules for known
benchmark domains.

No snapshot baseline command was run. No benchmark-regression override was used.

## Direct Dyld Output

`ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`

```text
Test project /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/build/zig
    Start 1: emel_tests_model_and_batch
1/1 Test #1: emel_tests_model_and_batch .......Subprocess aborted***Exception:   0.09 sec
dyld[33081]: dyld cache '(null)' not loaded: syscall to map cache into shared region failed
dyld[33081]: Library not loaded: /usr/lib/libSystem.B.dylib
  Referenced from: <C894ECC8-3470-3075-9E93-94D5E735D7E5> /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/build/zig/emel_tests_bin
  Reason: tried: '/usr/lib/libSystem.B.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/usr/lib/libSystem.B.dylib' (no such file), '/usr/lib/libSystem.B.dylib' (no such file, no dyld cache)

0% tests passed, 1 tests failed out of 1
```

`ctest --test-dir build/zig --output-on-failure -R 'emel_tests_(model_and_batch|io)'`

```text
Test project /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/build/zig
    Start 1: emel_tests_model_and_batch
1/2 Test #1: emel_tests_model_and_batch .......Subprocess aborted***Exception:   0.09 sec
dyld[35545]: dyld cache '(null)' not loaded: syscall to map cache into shared region failed
dyld[35545]: Library not loaded: /usr/lib/libSystem.B.dylib
  Referenced from: <C894ECC8-3470-3075-9E93-94D5E735D7E5> /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/build/zig/emel_tests_bin
  Reason: tried: '/usr/lib/libSystem.B.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/usr/lib/libSystem.B.dylib' (no such file), '/usr/lib/libSystem.B.dylib' (no such file, no dyld cache)

    Start 2: emel_tests_io
2/2 Test #2: emel_tests_io ....................Subprocess aborted***Exception:   0.00 sec
dyld[35616]: dyld cache '(null)' not loaded: syscall to map cache into shared region failed
dyld[35616]: Library not loaded: /usr/lib/libSystem.B.dylib
  Referenced from: <C894ECC8-3470-3075-9E93-94D5E735D7E5> /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/build/zig/emel_tests_bin
  Reason: tried: '/usr/lib/libSystem.B.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/usr/lib/libSystem.B.dylib' (no such file), '/usr/lib/libSystem.B.dylib' (no such file, no dyld cache)

0% tests passed, 2 tests failed out of 2
```

## Automated Substitute Evidence

The changed-file quality gate rebuilt and executed the focused shards under
`build/coverage`:

```text
running coverage test regex: ^emel_tests_(io|model_and_batch)$
Test project /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/build/coverage
    Start 1: emel_tests_model_and_batch
1/2 Test #1: emel_tests_model_and_batch .......   Passed    1.74 sec
    Start 2: emel_tests_io
2/2 Test #2: emel_tests_io ....................   Passed    0.10 sec

100% tests passed, 0 tests failed out of 2
```

Changed-file coverage passed with 94.8% line coverage and 66.5% branch coverage across
the scoped Phase 225 source files.

The same gate passed:

- generation benchmark suite
- diarization_sortformer benchmark suite
- paritychecker tests
- docs generation check
- fuzz smoke skip for no fuzz-affecting changed files

## Source Scans

Command:

```sh
rg -n "effect_dispatch_io_loads|for \\(.*io_loader->process_event|emel/io/read/detail.hpp|emel/io/read/events.hpp|read_tensor_request" src/emel/model/loader tools/bench/generation_bench.cpp tools/bench/diarization/sortformer_fixture.hpp tools/embedded_size/emel_probe/main.cpp tools/paritychecker/parity_engines.cpp
```

Result: no matches.

Additional command:

```sh
rg -n "io_load_spans|emel::io::source::load_file_bytes|\\.used_io_strategy = ev.used_io_strategy" tools/bench/generation_bench.cpp tools/bench/diarization/sortformer_fixture.hpp tools/embedded_size/emel_probe/main.cpp tools/paritychecker/parity_engines.cpp
```

Result: all four maintained caller files contain request-owned `io_load_spans`, public
source loading through `emel::io::source::load_file_bytes`, and public
`used_io_strategy` evidence propagation.

## Consistency Warnings

`node .codex/get-shit-done/bin/gsd-tools.cjs validate consistency` exited 0 with 16
pre-existing warnings: Phases 212-224 archive directories are not present in active
`.planning/phases`, Phase 211 exists on disk but is not in active `ROADMAP.md`, and there
is a numbering gap from 211 to 225. No Phase 225 validation or path-truth error was
reported.
