#pragma once

#include <array>

#include "emel/diarization/sortformer/cache/detail.hpp"
#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/diarization/sortformer/transformer/detail.hpp"

namespace emel::diarization::sortformer::executor::action {

struct context {
  context() {
    encoder_projection_weight_cache.lhs_4row.resize(
        emel::diarization::sortformer::modules::detail::
            k_encoder_projection_prepared_weight_value_count);
  }

  emel::diarization::sortformer::modules::detail::contract modules = {};
  emel::diarization::sortformer::transformer::detail::contract transformer = {};
  emel::diarization::sortformer::cache::detail::state cache = {};
  emel::diarization::sortformer::transformer::detail::layer_workspace transformer_workspace = {};
  emel::diarization::sortformer::detail::dense_weight_cache
      encoder_projection_weight_cache = {};
  std::array<float,
             emel::diarization::sortformer::transformer::detail::k_max_frame_count *
                 emel::diarization::sortformer::transformer::detail::k_hidden_dim>
      hidden_a = {};
  std::array<float,
             emel::diarization::sortformer::transformer::detail::k_max_frame_count *
                 emel::diarization::sortformer::transformer::detail::k_hidden_dim>
      hidden_b = {};
};

}  // namespace emel::diarization::sortformer::executor::action
