# tools parity plan

phased plan for the parity checker tool and its integration into quality gates.

## phase 1: gate wiring and scaffolding
- [ ] quality gate: add `scripts/parity.sh` to `scripts/quality_gates.sh`.
- [ ] add `tools/parity/` with CMake fetch_content for the reference implementation.
- [ ] add `parity_runner` CLI with `--update` for snapshot regeneration.
- [ ] add `scripts/parity.sh` that builds and runs the parity tool.
- [ ] update `AGENTS.md` to allow reference linkage and identifiers in `tools/parity` only.

## phase 2: fixtures and snapshot format
- [ ] add GGUF fixture generator helpers in `tools/parity`.
- [ ] create fixtures for encoder models: BPE, WPM, UGM, SPM, RWKV, PLaMo-2.
- [ ] add `snapshots/parity/cases.json` with curated input strings per fixture.
- [ ] add `snapshots/parity/expected/*.json` outputs generated via `scripts/parity.sh --update`.

## phase 3: encoder parity implementation
- [ ] load GGUF fixture into EMEL vocab.
- [ ] run encoder SM per fixture/input and capture tokens/count.
- [ ] run reference tokenize for the same input with `add_special=false` and
  `parse_special=false`.
- [ ] compare token counts and token IDs with exact matching.
- [ ] emit clear diffs: fixture, input index, first mismatch details.

## phase 4: gate enforcement and documentation
- [ ] document usage and fixture format in `docs/plans/tools-parity.plan.md`.
- [ ] ensure `scripts/parity.sh` is deterministic and fails on mismatch.
- [ ] quality gate: verify parity tool runs in `scripts/quality_gates.sh` and fails on diffs.
