---
created: 2026-04-02T02:05:39Z
title: Optimize LFM2.5 q6 prefill kernel
area: general
files:
  - src/emel/kernel/aarch64/actions.hpp:2960
  - src/emel/kernel/detail.hpp:744
  - src/emel/kernel/detail.hpp:981
  - tmp/profiles/lfm2_emel_parity_generation_sample.txt
  - tmp/profiles/lfm2_reference_generation_runtime_sample.txt
---

## Problem

The current EMEL LFM2.5 sample still shows prepared q6 compute immediately behind the q4 path:
`execute_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_unchecked(...)` and the argmax prepared q6
path are still prominent. Reference sampling shows q6 remains the second major quantized leaf too,
so EMEL still has ROI left in q6 after the initializer and q8-reuse work land.

## Solution

Treat q6 as the second prefill optimization wave. After removing request-path prep and redundant q8
construction, reprofile and optimize the prepared q6 i8mm path and matching argmax path until the
remaining short-run prefill gap narrows further.
