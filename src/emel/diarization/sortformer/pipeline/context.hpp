#pragma once

#include <array>

#include "emel/diarization/request/sm.hpp"
#include "emel/diarization/sortformer/encoder/detail.hpp"
#include "emel/diarization/sortformer/executor/sm.hpp"
#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/diarization/sortformer/pipeline/detail.hpp"

namespace emel::diarization::sortformer::pipeline::action {

struct context {
  emel::diarization::request::sm request = {};
  emel::diarization::sortformer::executor::sm executor = {};
  emel::diarization::sortformer::encoder::detail::contract encoder = {};
  emel::diarization::sortformer::modules::detail::contract modules = {};
  emel::diarization::sortformer::encoder::detail::pre_encoder_workspace encoder_workspace = {};
  std::array<float, detail::k_required_feature_count> features = {};
  std::array<float, detail::k_required_encoder_value_count> encoder_frames = {};
  std::array<float, detail::k_required_hidden_value_count> hidden = {};
};

}  // namespace emel::diarization::sortformer::pipeline::action
