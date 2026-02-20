#pragma once

#include <array>
#include <cstdint>

#include "emel/emel.h"
#include "emel/tensor/lifetime_analyzer/events.hpp"

namespace emel::tensor::lifetime_analyzer::action {

inline constexpr int32_t k_max_tensors = 2048;

struct context {
  const event::tensor_desc * tensors = nullptr;
  int32_t tensor_count = 0;
  int32_t ranges_out_count = 0;
  bool has_first_out = false;
  bool has_last_out = false;

  std::array<int32_t, k_max_tensors> tensor_ids = {};
  std::array<int32_t, k_max_tensors> first_use = {};
  std::array<int32_t, k_max_tensors> last_use = {};
  std::array<int32_t, k_max_tensors> n_children = {};
  std::array<int32_t, k_max_tensors> n_views = {};
  std::array<int32_t, k_max_tensors> view_src_indices = {};
  std::array<bool, k_max_tensors> tensor_is_view = {};
  std::array<bool, k_max_tensors> tensor_is_exec_node = {};
  std::array<bool, k_max_tensors> tensor_is_control_dep = {};

  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::tensor::lifetime_analyzer::action
