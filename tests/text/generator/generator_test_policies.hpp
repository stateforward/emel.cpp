#pragma once

#include "emel/text/generator/events.hpp"

namespace emel::text::generator::test {

inline constexpr emel::text::generator::route_policy k_generation_route_policy{
    .parallel_min_prefill_tokens = 8,
    .parallel_min_gemv_dim = 1024,
    .prefill_chunk4_min_tokens = emel::text::generator::k_prefill_q8_chunk_rows,
    .prefill_chunk8_min_tokens =
        emel::text::generator::k_prefill_q8_chunk8_rows,
};

inline emel::text::generator::runtime_policy
make_auto_runtime_policy(const emel::model::data &model) noexcept {
  return emel::text::generator::make_auto_runtime_policy(
      model, k_generation_route_policy);
}

} // namespace emel::text::generator::test
