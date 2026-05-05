# Phase 208 Context: Public Runtime And Evidence Surfaces

## Goals
Ensure that `model/loader`, benchmarking tools, paritycheckers, and embedded probes use only public runtime surfaces (e.g., IO and Tensor public events/contexts) for mmap strategy. Eliminate any actor-internal reach-through or tool-only mmap logic. Ensure benchmark and parity tools report accurate usage of mmap, indicating "unsupported" or "non-mmap" where applicable instead of faking mmap parity.

## Requirements
- **TIO-03**: `model/loader`, maintained benchmark lanes, paritychecker lanes, and embedded probes can select or report mmap-backed loading only through public runtime surfaces, with no low-level mmap logic or actor-internal reach-through.
- **VAL-04**: Maintained benchmark and parity evidence reports mmap usage only when the EMEL lane actually runs the mmap-backed runtime path and does not present unsupported fallback behavior as mmap strategy parity or performance.

## SML Rules Context
According to `AGENTS.md` and `docs/rules/sml.rules.md`:
- `process_event` must not be bypassed; cross-machine/actor interaction must occur through explicit events.
- Actor contexts must not be read or mutated directly from outside; state inspection is done via `visit_current_states` or `is(...)`.
- We must not use test-only control fields to skip the E2E flow.
- We must not emulate mmap or present fallback behavior as mmap.

## Scope
- `model/loader`
- Maintained benchmark tools (`tools/bench/*`, `scripts/bench*.sh`)
- Paritychecker lanes (`tools/paritychecker/*`)
- Embedded probes
- NO internal actor reach-through.
- NO tool-only scaffolds or low-level mmap outside the IO strategy bounded domain.
