# boost.SML benchmark notes

## dispatch overhead (sml vs naive)

Built and ran:
- `benchmark/connection/boost_sml.cpp`
- `benchmark/connection/naive_if_else.cpp`
- `benchmark/connection/naive_switch.cpp`

with `--benchmark_min_time=0.5s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`

## results (real_time, ns/op)
- `naive_if_else`: mean 6.605687, median 6.432878
- `naive_switch`: mean 6.762860, median 6.779858
- `boost_sml`: mean 6.617205, median 6.617472

## relative to `naive_if_else`
- `boost_sml`: +0.174% mean, +2.870% median
- `naive_switch`: +2.379% mean, +5.394% median

## takeaway
In this benchmark, `sml::process_event` is effectively at parity with direct function-call dispatch (`naive_if_else`) on this machine, with low single-digit overhead at most.

Absolute throughput is around ~150M events/sec (1e9 / ~6.6ns).

*Raw outputs are in:*
- `@../../../../../../tmp/sml_bench/boost_sml.json`
- `@../../../../../../tmp/sml_bench/naive_if_else.json`
- `@../../../../../../tmp/sml_bench/naive_switch.json`
