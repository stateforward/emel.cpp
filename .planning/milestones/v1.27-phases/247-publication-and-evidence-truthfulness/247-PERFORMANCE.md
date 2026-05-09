# Phase 247 Loading Strategy Performance Evidence

## Maintained Benchmark Path

The maintained benchmark evidence uses the public generation benchmark and the shared
`EMEL_MODEL_LOAD_IO_STRATEGY` strategy selector:

```sh
EMEL_MODEL_LOAD_IO_STRATEGY=<strategy> scripts/bench.sh --snapshot --compare --suite=generation
```

This measures the maintained generation entrypoint, not an isolated byte-copy microbenchmark.
Therefore the numbers below are end-to-end generation benchmark observations with strategy
selection/reporting active. They are useful for detecting maintained entrypoint regressions and for
proving which strategy path is actually accepted, but they should not be quoted as pure kernel or
I/O-only throughput.

## Observed Results

| Strategy request | Maintained path result | EMEL observed time | Reference observed time | Notes |
|------------------|------------------------|--------------------|-------------------------|-------|
| `none` | accepted baseline | 433586042 ns/op | 326626417 ns/op | baseline public generation run |
| `read_copy` | accepted | 420805292 ns/op | 316159083 ns/op | storage-backed public loader path accepted |
| `staged_read` | accepted | 439415042 ns/op | 315079375 ns/op | staged public loader path accepted |
| `mapped_file` | unsupported in this maintained path | n/a | n/a | direct tensor mmap exists, but generation strategy selection does not publish a maintained mmap run here |
| `cooperative_async` | accepted after Phase 248 | 488169958 ns/op | 353585500 ns/op | maintained generation path reaches public `io/loader` async route; observed ratio 1.381x |

## Closeout Blocker

Resolved by Phase 248. Truthfully reporting `cooperative_async` as unsupported prevented a false
performance claim during Phase 247, but it did not satisfy `PERF-01`. Phase 248 wired the
maintained model-loader/generation path through the public async route and recorded a measured
`cooperative_async` result.

## Interpretation

- `read_copy` was the fastest accepted maintained strategy in this single observed generation run
  (`420805292 ns/op`), roughly 3% faster than `none` and 4% faster than `staged_read`.
- `staged_read` was accepted but slightly slower than the baseline in this run. The difference is
  small relative to full generation runtime and should be treated as entrypoint-level evidence, not
  isolated I/O throughput.
- `cooperative_async` now executes end to end in the maintained generation path. The observed run is
  slower than the baseline/reference comparison on this fixture, but it proves the maintained async
  route and provides honest measured evidence instead of an unsupported-path placeholder.
- `mapped_file` remains a direct tensor-owned strategy rather than a maintained generation strategy
  selection result in this benchmark path.
