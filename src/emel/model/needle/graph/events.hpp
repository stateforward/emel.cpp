#pragma once

#include <cstdint>
#include <span>

#include "emel/error/error.hpp"
#include "emel/model/needle/events.hpp"
#include "emel/model/needle/graph/errors.hpp"

namespace emel::model::needle::graph::event {

// Binds the named contract baked into the machine at construction: validates
// geometry against the graph capacities, decodes the fp16 norm/gate scales to
// f32 once, precomputes the RoPE tables, and clears the KV/engram state. No
// allocation happens here; all storage is sized at construction.
struct init {
  // Deployment is the default. Tests may explicitly request the legacy
  // W4/f32 parity route; the choice is copied into the init/step SML payload.
  bool activation_quant = true;
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
};

struct configure_cq_timing {
  bool enabled = false;
};

struct capture_cq_timing {
  uint64_t &nanoseconds;
};

struct capture_a8_diagnostics {
  uint64_t &quantize_calls;
};

struct init_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool activation_quant = true;
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
  bool activation_quant = true;
  emel::error::type err = emel::error::cast(error::none);
};

struct step_run {
  step_ctx &ctx;
};

} // namespace emel::model::needle::graph::event
