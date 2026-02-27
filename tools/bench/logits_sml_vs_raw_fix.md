# Problem Statement: Logits SML Architecture Performance Degradation

## Benchmark Findings
The recent benchmark conclusively isolates a massive execution latency inside the `logits` sub-system:

- **128k Vocab Size (SML):** `1,382,815 ns` (1.383 ms per token)
- **128k Vocab Size (Raw C++):** `92,138 ns` (0.092 ms per token)
- **Combined Inference Degradation (Validator + Sampler):** The SML architecture is operating roughly **~15x to ~18x** slower than plain C++ across all vocabulary scales (32k, 128k, 256k).

At massive vocabulary scale (256k), generating 500 tokens using the raw implementation takes about **0.11 seconds**. Generating the same 500 tokens using the SML `process_event` state-iteration takes almost **2.0 seconds** solely spent traversing states per token.

## Architectural Root Cause
The `boost.sml` implementation inside the `logits` system is being utilized improperly as a purely iterative loop. 
Currently, the State Machine models the *data traversal* across `vocab_size` through recurring anonymous transitions (`sml::completion<>`). 

SML is designed for macro-level state orchestration, not tight, inner-loop arithmetic. 
1. **Instruction Cache & Branch Prediction Thrashing:** For a 128k vocabulary, the system drops into `process_internal_events(...)` 128,000 times per token generation. 
2. **SIMD / Vectorization Breaking:** The compiler cannot autovectorize array mutations across state transitions. Operations that `SIMD` normally executes in blocks of 256-bits or 512-bits are reduced to serialized $O(1)$ branches. 

## Required Resolution (Hard Cutover)
The engineer MUST decouple data iteration from the state machine logic in `src/emel/logits/` entirely.

1. **Macro State, Not Micro Data:** The `logits/validator/sm.hpp` and `logits/sampler/sm.hpp` state graphs must only govern the *lifecycle* of the generation algorithm (e.g. `Ready` -> `Filtering` -> `Sampling` -> `Done`), not the iteration over vocabulary entries.
2. **Lift the Loops to Actions:** Move the $O(N)$ candidate iteration loops completely inside a single, isolated execution block within `actions.hpp` (e.g., `action::apply_penalties{}`).
3. **No SML in the Hot Path:** Ban `sml::completion<>` for any inner-loop iterations inside inference kernels.
