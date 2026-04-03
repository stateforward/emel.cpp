# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.0-ROADMAP.md) - shipped 2026-03-08 with 7 phases and 15 plans; proved one canonical Llama-68M generation parity slice in `tools/paritychecker/`.
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.1-ROADMAP.md) - shipped 2026-03-11 with 4 phases and 10 plans; added one truthful canonical Llama-68M generation benchmark in `tools/bench`, native EMEL decode benchmarking, compare output, and snapshot/docs integration.
- [x] [v1.2: Flash Attention](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.2-ROADMAP.md) - shipped 2026-03-22 with 5 phases and 13 plans; added an EMEL-owned flash-attention path to the canonical Llama-68M slice, hard-cut runtime tensor lifecycle through `emel::tensor::sm`, and published maintained benchmark evidence over a preserved pre-flash baseline.
- [x] [v1.3: ARM Flash Optimizations](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.3-ROADMAP.md) - shipped 2026-03-22 with 3 phases and 7 plans; delivered optimized AArch64 flash execution, maintained runtime/parity attribution, and preserved-baseline benchmark publication for the canonical ARM Llama-68M slice.
- [x] [v1.4: Full Vectorized Quantized Kernels](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.4-ROADMAP.md) - shipped 2026-03-25 with 5 phases and 11 plans; delivered EMEL-owned vectorized q2/q3/q6 kernels, full maintained `1/10/100/1000` parity proof, and quantized benchmark attribution on the canonical ARM slice.
- [x] [v1.5: Full ARM Quantized Path](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.5-ROADMAP.md) - shipped 2026-03-27 with 5 phases and 10 plans; closed the maintained ARM quantized-path contract and restored canonical flash publication.
- [x] [v1.6: Qwen3-0.6B Parity And Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.6-ROADMAP.md) - shipped 2026-03-30 with 5 phases and 12 plans; brought one canonical Qwen3 slice up through the maintained generator, parity, and benchmark surfaces.
- [x] [v1.7: Generator Prefill Submachine Decomposition](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.7-ROADMAP.md) - shipped 2026-03-30 with 3 phases and 6 plans; extracted generator-owned prefill orchestration while preserving maintained proof.
- [x] [v1.8: Truthful Qwen3 E2E Embedded Size](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.8-ROADMAP.md) - shipped 2026-04-02 with 6 phases and 8 plans; published one truthful maintained Qwen3 executable-size comparison with smoke proof and generated README evidence.
- [x] [v1.9: Liquid LFM2.5-1.2B Thinking ARM Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.9-ROADMAP.md) - shipped 2026-04-02 with 8 phases and 9 plans; delivered one maintained Liquid `lfm2` ARM slice with explicit runtime support, additive parity coverage, benchmark publication, and audit-backed archive proof.

## Current Milestone

No active milestone.

Start the next milestone with `$gsd-new-milestone`.
