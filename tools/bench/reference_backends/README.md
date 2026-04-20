# Reference Backends

Each manifest in this directory describes one pluggable reference backend for
`tools/bench/embedding_compare.py`.

## Contract

- `id`: stable backend selector used by `--reference-backend`
- `display_name`: operator-facing label
- `language`: backend implementation language, currently `python` or `cpp`
- `description`: concise purpose/provenance note
- `build_command`: optional repo-relative command array run before the backend
  executes
- `run_command`: required repo-relative command array that emits
  `embedding_compare/v1` JSONL records

## Isolation

The manifest only describes the reference lane. The EMEL lane remains separate
and continues to run through `embedding_generator_bench_runner`.
