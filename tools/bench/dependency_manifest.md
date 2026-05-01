# Benchmark Dependency Manifest

`bench_dependency_manifest/v1` is a deterministic line-oriented manifest emitted by
`bench_runner` for benchmark gate impact detection.

Generate or refresh the baseline with:

```sh
build/bench_tools_ninja/bench_runner --write-dependency-manifest \
  tools/bench/dependency_manifest.txt
```

The schema starts with:

```text
bench_dependency_manifest/v1
full_gate_on=missing,stale,uncertain
```

Each `record` line includes:

- `runner`: a benchmark runner suite name, or `all` for shared benchmark infrastructure.
- `kind`: `source`, `config`, `fixture`, `model`, `script`, or `snapshot`.
- `path`: repo-relative source, config, fixture, model, script, or snapshot input.
- `reason`: the conservative reason the input affects the runner.

Check freshness with:

```sh
build/bench_tools_ninja/bench_runner --check-dependency-manifest \
  tools/bench/dependency_manifest.txt
```

Fresh output reports `full_gate=0 reason=fresh`. Missing, stale, or uncertain manifest data is
reported as `full_gate=1` and must force the relevant benchmark gate or a full benchmark gate.
