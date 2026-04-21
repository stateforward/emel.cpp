# Embedding Variants

Each `*.json` file defines one maintained embedding benchmark variant for the shared
`embedding_compare/v1` workflow.

Adding an ordinary maintained variant should be data-only:

1. Add one manifest in this directory.
2. Reuse an existing `payload_id` unless the benchmark runner needs a genuinely new input payload.
3. Keep `id`, `case_name`, and `compare_group` stable because compare summaries key on them.

Do not edit benchmark runner, compare, or test enumeration code for ordinary additions that reuse
existing payloads.
