---
created: 2026-04-02T02:05:39Z
title: Optimize LFM2.5 q4 prefill kernel
area: general
files:
  - src/emel/kernel/aarch64/actions.hpp:1942
  - src/emel/kernel/detail.hpp:626
  - src/emel/kernel/detail.hpp:934
  - tmp/profiles/lfm2_emel_parity_generation_sample.txt
  - tmp/profiles/lfm2_reference_generation_runtime_sample.txt
---

## Problem

After the current q4/q6 improvements, the clean EMEL runtime sample still shows
`dot_q4_k_x8_q8_k_group_bl8_neon(...)` as the top prefill leaf. Reference sampling shows the same
effective workload is also q4-dominated on the ggml side, but EMEL still pays a larger prefill cost
on the short LFM2.5 runs.

## Solution

Keep the EMEL lane native and optimize the q4 BL8 prefill path directly. Prioritize operand access,
accumulation, and any remaining q4 pack/setup overhead that still appears on the runtime sample.
Measure the `1` and `10` token LFM2.5 prefill delta after each kernel pass.
