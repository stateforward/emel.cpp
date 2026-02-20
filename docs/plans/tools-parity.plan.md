# Tools parity plan

Phased plan for the parity checker tool and its integration into quality gates.

## Phase 1: Gate wiring and scaffolding
- [ ] Quality gate: add `scripts/parity.sh` to `scripts/quality_gates.sh`.
- [ ] Add `tools/parity/` with CMake FetchContent for the reference implementation.
- [ ] Add `parity_runner` CLI with `--update` for snapshot regeneration.
- [ ] Add `scripts/parity.sh` that builds and runs the parity tool.
- [ ] Update `AGENTS.md` to allow reference linkage and identifiers in `tools/parity` only.

## Phase 2: Fixtures and snapshot format
- [ ] Add GGUF fixture generator helpers in `tools/parity`.
- [ ] Create fixtures for encoder models: BPE, WPM, UGM, SPM, RWKV, PLaMo-2.
- [ ] Add `snapshots/parity/cases.json` with curated input strings per fixture.
- [ ] Add `snapshots/parity/expected/*.json` outputs generated via `scripts/parity.sh --update`.

## Phase 3: Encoder parity implementation
- [ ] Load GGUF fixture into EMEL vocab.
- [ ] Run encoder SM per fixture/input and capture tokens/count.
- [ ] Run reference tokenize for the same input with `add_special=false` and
  `parse_special=false`.
- [ ] Compare token counts and token IDs with exact matching.
- [ ] Emit clear diffs: fixture, input index, first mismatch details.

## Phase 4: Gate enforcement and documentation
- [ ] Document usage and fixture format in `docs/plans/tools-parity.plan.md`.
- [ ] Ensure `scripts/parity.sh` is deterministic and fails on mismatch.
- [ ] Quality gate: verify parity tool runs in `scripts/quality_gates.sh` and fails on diffs.
