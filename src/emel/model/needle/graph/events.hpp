#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "emel/error/error.hpp"
#include "emel/kernel/cq/events.hpp"
#include "emel/model/needle/events.hpp"
#include "emel/model/needle/graph/errors.hpp"

namespace emel::model::needle::graph::event {

// Binds the named contract baked into the machine at construction: validates
// geometry against the graph capacities, decodes the fp16 norm/gate scales to
// f32 once, precomputes the RoPE tables, and clears the KV/engram state. No
// allocation happens here; all storage is sized at construction.
struct init {
  // The maintained heldout reference uses CQ4 weights with f32 activations.
  // Pass true explicitly for the training/A8 parity route.
  bool activation_quant = false;
};

// Runs the prompt through the graph: one allocation-free RTC step dispatch per
// token. `logits_out` receives the final position's logits (vocab floats).
struct prefill {
  std::span<const int32_t> tokens = {};
  std::span<float> logits_out = {};
};

// Runs one decode step for `token` at the next cache position and writes the
// logits (vocab floats).
struct decode {
  int32_t token = 0;
  std::span<float> logits_out = {};
};

// Snapshot-style diagnostics for tests/bench attribution. Observation is
// allocation-free and does not drive machine progression.
struct capture_cq_diagnostics {
  uint64_t &prepare_calls;
  uint64_t &prepared_calls;
  size_t &prepared_index_bytes;
  size_t &prepared_input32_bytes;
  size_t &prepared_norm_bytes;
  size_t &prepared_group32_norm_bytes;
};

struct capture_projection_diagnostics {
  std::array<uint64_t, 3u> &worker_calls;
  uint64_t &submitted;
  uint64_t &joined;
  uint64_t &live;
};

struct capture_swa_diagnostics {
  uint64_t &gqa2_calls;
};

struct configure_cq_timing {
  bool enabled = false;
  emel::kernel::cq::event::timestamp_now_fn now = nullptr;
};

struct capture_cq_timing {
  emel::kernel::cq::event::timing_breakdown &breakdown;
};
using timestamp_now_fn = emel::kernel::cq::event::timestamp_now_fn;

// Opt-in graph/component timing. The graph calls `now` only while enabled;
// capture and reset are observation/control events and never advance runtime.
struct timing_breakdown {
  uint64_t steps = 0u;
  uint64_t total_nanoseconds = 0u;
  uint64_t cq_nanoseconds = 0u;
  uint64_t graph_overhead_nanoseconds = 0u;
  uint64_t engram_nanoseconds = 0u;
  uint64_t norm_nanoseconds = 0u;
  uint64_t mhc_pre_nanoseconds = 0u;
  uint64_t mhc_post_nanoseconds = 0u;
  uint64_t attention_rope_nanoseconds = 0u;
  uint64_t attention_cache_nanoseconds = 0u;
  uint64_t attention_attend_nanoseconds = 0u;
  uint64_t attention_gate_nanoseconds = 0u;
  uint64_t hadamard_nanoseconds = 0u;
  uint64_t lane_copy_mean_nanoseconds = 0u;
  uint64_t sampling_nanoseconds = 0u;
};

struct configure_timing {
  bool enabled = false;
  timestamp_now_fn now = nullptr;
};

struct reset_timing {};

struct capture_timing {
  timing_breakdown &breakdown;
};

struct capture_a8_diagnostics {
  uint64_t &quantize_calls;
};

struct init_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool activation_quant = false;
};

struct init_run {
  const init &request;
  init_ctx &ctx;
};

// Internal per-step runtime payload (mutable, never publicly exposed):
// carries the dispatch-local step data across the completion chain.
struct step_ctx {
  int32_t token = 0;
  uint32_t position = 0u;
  uint32_t layer_index = 0u;
  bool want_logits = false;
  std::span<float> logits_out = {};
  bool activation_quant = false;
  emel::error::type err = emel::error::cast(error::none);
};

struct step_run {
  step_ctx &ctx;
};

} // namespace emel::model::needle::graph::event
