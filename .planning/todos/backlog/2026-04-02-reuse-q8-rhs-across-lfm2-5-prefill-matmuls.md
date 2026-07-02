---
created: 2026-04-02T02:05:39Z
title: Reuse q8 RHS across LFM2.5 prefill matmuls
area: general
files:
  - src/emel/generator/detail.hpp:1660
  - src/emel/generator/detail.hpp:1792
  - src/emel/generator/detail.hpp:1971
  - tmp/profiles/lfm2_emel_parity_generation_sample.txt
---

## Problem

The LFM2.5 EMEL runtime sample shows `matmul_vector_q8_input(...)` as the dominant generator-side
caller under `run_layer(...)`. That suggests the same normalized activation may still be quantized
and rebuilt multiple times within one layer phase for q/k/v, FFN, and argmax consumers.

## Solution

Audit the prefill path for repeated q8-RHS construction from the same input span. Hoist and reuse
the q8 representation where the source activation is identical and the reuse does not violate RTC
or domain boundaries. Reprofile after the reuse lands to confirm the redundant q8 work disappears
from the top stack.
