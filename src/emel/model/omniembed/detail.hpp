#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/detail.hpp"

namespace emel::model::omniembed::detail {

struct tensor_view {
  const emel::model::data::tensor_record * tensor = nullptr;
  std::string_view name = {};
};

struct family_view {
  std::string_view prefix = {};
  uint32_t tensor_count = 0;
  tensor_view first = {};
};

struct execution_contract {
  const emel::model::data * model = nullptr;
  int32_t embedding_length = 0;
  int32_t image_encoder_length = 0;
  int32_t audio_encoder_length = 0;
  uint32_t matryoshka_dimension_count = 0;
  std::array<int32_t, emel::model::data::k_max_matryoshka_dims> matryoshka_dimensions = {};
  family_view text_encoder = {};
  family_view text_projection = {};
  family_view image_encoder = {};
  family_view image_projection = {};
  family_view audio_encoder = {};
  family_view audio_projection = {};
};

bool is_execution_architecture(std::string_view architecture) noexcept;

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept;

emel::error::type build_execution_contract(const emel::model::data & model_data,
                                           execution_contract & contract_out) noexcept;

emel::error::type validate_data(const emel::model::data & model_data) noexcept;

emel::error::type validate_execution_contract(const emel::model::data & model_data) noexcept;

}  // namespace emel::model::omniembed::detail
