# Embedding Variants

Each `*.json` file defines one maintained embedding benchmark variant for the shared
`embedding_compare/v1` workflow. Variants are discovered recursively below isolation
subdirectories; do not place manifests directly in this directory.

Adding an ordinary maintained variant should be data-only:

1. Add one manifest under a grouped subdirectory such as
   `tools/bench/embedding_variants/<model-family>/<modality>/`.
2. Reuse an existing `payload_id` unless the benchmark runner needs a genuinely new input payload.
3. Keep `id`, `case_name`, and `compare_group` stable because compare summaries key on them.

Do not edit benchmark runner, compare, or test enumeration code for ordinary additions that reuse
existing payloads.

Current layout:

- `te75m/text/`
- `te75m/image/`
- `te75m/audio/`
