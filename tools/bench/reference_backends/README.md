# Reference Backends

Each manifest in this directory describes one pluggable reference backend for a
repo-owned compare surface such as `tools/bench/embedding_compare.py` or
`tools/bench/generation_compare.py`.

## Contract

- `id`: stable backend selector used by `--reference-backend`
- `surface`: compare-surface schema that owns the manifest, such as
  `embedding_compare/v1` or `generation_compare/v1`
- `display_name`: operator-facing label
- `language`: backend implementation language, currently `python` or `cpp`
- `description`: concise purpose/provenance note
- `build_command`: optional repo-relative command array run before the backend
  executes
- `run_command`: required repo-relative command array that emits
  compare-surface JSONL records for the selected `surface`
- `supports_exact_variant_id`: optional `true` for embedding backends that honor
  `EMEL_BENCH_VARIANT_ID` as an exact manifest-ID selector

## Isolation

The manifest only describes the reference lane. The EMEL lane remains separate
and continues to run through its existing repo-owned runner surface.

## Variant Selection

Generation compare uses `--workload-id` to select one discovered generation workload manifest.
Embedding compare uses `--variant-id` to select one discovered embedding variant manifest by
exact ID. The older `--case-filter` remains available for broad substring filtering, but
maintained data-only variant additions should prefer the stable manifest IDs. Backends that do
not declare `supports_exact_variant_id` reject `--variant-id` instead of mixing exact EMEL
selection with broad or unsupported reference selection.
