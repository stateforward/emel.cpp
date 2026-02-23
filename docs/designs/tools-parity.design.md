---
title: tools parity design
status: draft
---

# tools parity design

phased design outline for the parity checker tool and its integration into quality gates.

## phase 1: gate wiring and scaffolding
- [ ] quality gate: add `scripts/paritychecker.sh` to `scripts/quality_gates.sh`.
- [ ] add `tools/paritychecker/` with CMake fetch_content for the reference implementation.
- [ ] add `parity_runner` CLI snapshot-update mode for baseline regeneration.
- [ ] add `scripts/paritychecker.sh` that builds and runs the parity tool.
- [ ] update `AGENTS.md` to allow reference linkage and identifiers in `tools/paritychecker` only.

## phase 2: fixtures and snapshot format
- [ ] add GGUF fixture generator helpers in `tools/paritychecker`.
- [ ] create fixtures for encoder models: BPE, WPM, UGM, SPM, RWKV, PLaMo-2.
- [ ] add `snapshots/paritychecker/cases.json` with curated input strings per fixture.
- [ ] add `snapshots/paritychecker/expected/*.json` outputs generated via paritychecker snapshot-update mode.

## phase 3: encoder parity implementation
- [ ] load GGUF fixture into EMEL vocab.
- [ ] run encoder SM per fixture/input and capture tokens/count.
- [ ] run reference tokenize for the same input with `add_special=false` and
  `parse_special=false`.
- [ ] compare token counts and token IDs with exact matching.
- [ ] emit clear diffs: fixture, input index, first mismatch details.

## phase 4: gate enforcement and documentation
- [ ] document usage and fixture format in `docs/designs/tools-parity.design.md`.
- [ ] ensure `scripts/paritychecker.sh` is deterministic and fails on mismatch.
- [ ] quality gate: verify parity tool runs in `scripts/quality_gates.sh` and fails on diffs.
