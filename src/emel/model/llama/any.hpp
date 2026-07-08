#pragma once

#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/transformer/any.hpp"

namespace emel::model::llama {

using block_view = emel::model::transformer::block_view;
using execution_view = emel::model::transformer::execution_view;
using quantized_contract_kind = emel::model::transformer::quantized_contract_kind;
using quantized_path_audit = emel::model::transformer::quantized_path_audit;
using quantized_stage_audit = emel::model::transformer::quantized_stage_audit;
using quantized_stage_family = emel::model::transformer::quantized_stage_family;

inline emel::error::type build_execution_view(const emel::model::data & model_data,
                                              execution_view & view_out) noexcept {
  return emel::model::transformer::build_execution_view(model_data, view_out);
}

inline emel::error::type lookup_block_view(const execution_view & execution,
                                           const int32_t block_index,
                                           block_view & block_out) noexcept {
  return emel::model::transformer::lookup_block_view(execution, block_index, block_out);
}

inline quantized_path_audit build_quantized_path_audit(
    const execution_view & execution) noexcept {
  return emel::model::transformer::build_quantized_path_audit(execution);
}

inline std::string_view quantized_stage_family_name(
    const quantized_stage_family family) noexcept {
  return emel::model::transformer::quantized_stage_family_name(family);
}

inline std::string_view quantized_contract_kind_name(
    const quantized_contract_kind kind) noexcept {
  return emel::model::transformer::quantized_contract_kind_name(kind);
}

inline std::string_view tensor_type_name(const int32_t tensor_type) noexcept {
  return emel::model::transformer::tensor_type_name(tensor_type);
}

}  // namespace emel::model::llama
