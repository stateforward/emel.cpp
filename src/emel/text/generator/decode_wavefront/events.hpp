#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/error/error.hpp"
#include "emel/graph/events.hpp"
#include "emel/kernel/any.hpp"
#include "emel/text/generator/decode_wavefront/errors.hpp"
#include "emel/text/generator/events.hpp"

namespace emel::graph {
struct sm;
}

namespace emel::text::generator::decode_wavefront::event {

inline constexpr size_t k_max_lanes = 8u;
inline constexpr int32_t k_no_failed_lane = -1;

enum class kernel_route : uint8_t {
  packed_q8_0,
  q8_k,
  native_quantized,
  native_quantized_q8_k_logits,
  kernel,
};

enum class output_contract : uint8_t {
  materialized_logits,
  preselected_argmax,
};

struct compatibility_key {
  const void * model_identity = nullptr;
  const void * backend_identity = nullptr;
  emel::kernel::kernel_kind kernel_kind = emel::kernel::kernel_kind::x86_64;
  emel::text::generator::attention_mode attention =
      emel::text::generator::attention_mode::nonflash;
  kernel_route route = kernel_route::kernel;
  output_contract output = output_contract::materialized_logits;
  uint32_t dtype_layout_contract = 0u;
  uint32_t quantized_contract = 0u;
  int32_t step_size = 1;
  int32_t token_count = 1;
};

struct dispatch_summary {
  emel::error::type err = emel::error::cast(error::none);
  bool grouped = false;
  int32_t dispatched_lanes = 0;
  int32_t failed_lane = k_no_failed_lane;
};

struct lane {
  lane(emel::graph::sm & graph_ref,
       emel::graph::event::compute & compute_ref,
       const compatibility_key key_ref,
       bool & accepted_ref) noexcept
    : graph(graph_ref), compute(compute_ref), key(key_ref), accepted(accepted_ref) {}

  emel::graph::sm & graph;
  emel::graph::event::compute & compute;
  compatibility_key key;
  bool & accepted;
};

struct run {
  run(std::span<lane> lanes_ref, dispatch_summary & out_ref) noexcept
    : lanes(lanes_ref), out(out_ref) {}

  std::span<lane> lanes = {};
  dispatch_summary & out;
};

}  // namespace emel::text::generator::decode_wavefront::event
