# Phase 235 Verification

All commands run from milestone worktree:
`/Users/gabrielwillen/.atmux/teams/emel_cpp/milestone63/worktree`

## Snapshot Status

- Milestone `snapshots/quality_gates/timing.txt`: clean
- Root `snapshots/quality_gates/timing.txt`: clean

## Validation Commands and Results

1. Build doctest binary

```bash
ninja -C build emel_tests_bin
```

Result: **PASS**

2. Targeted Phase 235 guardrail doctests (GRD-01/02/03)

```bash
./build/emel_tests_bin --test-case="model loader io boundary uses actor events without helper exposure,model_tensor_owns_staged_read_residency_boundary,phase 235 grd-03 staged scheduling has no coroutine scaffolding tokens" --no-intro
```

Result: **PASS**
`3 passed, 0 failed`

3. GRD-04 mmap semantics regression proof

```bash
./build/emel_tests_bin --test-case="io mmap returns a deterministic mapped descriptor on success,io mmap release happy path returns slot to the free pool" --no-intro
```

Result: **PASS**
`2 passed, 0 failed`

4. GRD-05 bulk io/read semantics regression proof

```bash
./build/emel_tests_bin --test-case="io loader read copy batch routes once through io read" --no-intro
```

Result: **PASS**
`1 passed, 0 failed`

5. Focused io/model ctest lane

```bash
ctest --test-dir build --output-on-failure -R 'emel_tests_(io|model)'
```

Result: **PASS**
- `emel_tests_model_and_batch`: pass
- `emel_tests_io`: pass

## Quality Gate

- Scoped quality gate attempted in milestone worktree for this fix set: **NO** (not attempted in this final milestone-only pass).
- No quality-gate pass claim is made in this verification record.

## Residual Risk

- Guardrails are source-token and ownership-scan based for GRD-01/02/03; they are deterministic but can miss future leak patterns if new syscall/coroutine idioms appear under different tokens.
- GRD-04/05 evidence is sourced from shipped mmap and bulk io/read suites (not staged-read subset reruns).
