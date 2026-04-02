---
created: 2026-04-02T02:05:39Z
title: Move eager quant prepack into generator initializer
area: general
files:
  - src/emel/generator/detail.hpp:989
  - src/emel/generator/detail.hpp:1191
  - src/emel/kernel/detail.hpp:934
  - src/emel/kernel/detail.hpp:981
  - tmp/profiles/lfm2_emel_parity_generation_sample.txt
---

## Problem

LFM2.5 profiling shows EMEL still performs static q4/q6 prepared-layout work during generation.
The current hot path still reaches `prepare_native_matrix_layout(...)`, `pack_q4_k_rows_x8(...)`,
`make_block_q4_k_x8(...)`, and the prepared q6 packers while servicing one request. That inflates
prefill cost and is the wrong ownership boundary for static model-weight transforms.

## Solution

Move q4/q6 prepared-layout construction into the generator initializer submachine so it runs once
per model initialization. Keep the prepared layouts in EMEL-owned backend state and remove lazy
prepack from the request path. `generate()` should only consume prepared layouts.
