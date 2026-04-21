# Generation Workload Manifests

These manifests pin the maintained generation benchmark workloads that feed the shared
`generation_compare/v1` output contract.

Generation workloads are discovered recursively from `*.json` files below isolation
subdirectories. Do not place workload manifests directly in this directory; group them by model
family, prompt shape, and comparison lane so ownership stays local as the registry grows.

Ordinary workload additions should not edit `tools/bench/generation_bench.cpp`,
`tools/bench/generation_compare.py`, or benchmark test enumeration code.

To add a workload:

1. Add or reuse a prompt fixture in `tools/bench/generation_prompts/`.
2. Add one workload manifest under a grouped subdirectory such as
   `tools/bench/generation_workloads/<model-family>/<prompt-shape>/<lane>/`.
3. Give the manifest a stable `id`, `case_name`, and `compare_group`.
4. Use `--workload-id <id>` to run only that workload through
   `scripts/bench_generation_compare.sh`.

Current layout:

- `gemma4/single_user_hello/`
- `lfm2/single_user_hello/parity/`
- `lfm2/single_user_hello/single_lane/`
- `qwen3/single_user_hello/`

## Contract

- `schema`: workload manifest schema version
- `id`: stable workload identifier
- `case_name`: benchmark row name
- `compare_group`: stable matching group for EMEL/reference records
- `fixture_name`: operator-facing model fixture name
- `fixture_rel`: repo-relative model fixture path
- `fixture_slug`: stable fixture slug used in case naming
- `prompt_fixture_id`: expected checked-in prompt fixture id
- `prompt_fixture_path`: repo-relative prompt fixture path
- `formatter_mode`: stable formatter mode identifier
- `formatter_contract`: exact maintained formatter contract string
- `sampling_id`: stable sampling contract id
- `stop_id`: stable stop-condition contract id
- `seed`: deterministic sampling seed
- `max_output_tokens`: token budget pinned for the workload
- `comparable`: whether the workload may participate in EMEL/reference parity comparison
- `comparison_mode`: truthful compare label for machine-readable records
- `comparability_note`: explicit reason published into compare records
- `current_publication`: whether the workload is the current maintained publication slice

## Single-Lane Publication Proof

`lfm2_single_user_hello_max_tokens_1_single_lane_v1` is a maintained local proof that the
operator-facing compare wrapper publishes intentional EMEL-only workloads as
`non_comparable`. It uses the same checked-in LFM2 fixture as the parity publication row, but its
manifest contract sets `comparison_mode` to `single_lane` and `comparable` to `false`, so the
reference backend is intentionally not run for that selected workload.
