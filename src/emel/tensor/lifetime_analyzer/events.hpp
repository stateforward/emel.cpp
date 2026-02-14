#pragma once

#include <array>
#include <cstdint>

namespace emel::tensor::lifetime_analyzer::event {

inline constexpr int32_t k_max_sources = 4;

struct tensor_desc {
  int32_t tensor_id = -1;
  std::array<int32_t, k_max_sources> src_ids = {{-1, -1, -1, -1}};
  bool is_view = false;
  int32_t view_src_id = -1;
  bool is_exec_node = true;
  bool is_control_dep = false;
};

struct analyze {
  const tensor_desc * tensors = nullptr;
  int32_t tensor_count = 0;
  int32_t * first_use_out = nullptr;
  int32_t * last_use_out = nullptr;
  int32_t ranges_out_count = 0;
  int32_t * error_out = nullptr;
};

struct validate {
  int32_t * error_out = nullptr;
};
struct collect_ranges {
  int32_t * error_out = nullptr;
};
struct publish {
  int32_t * error_out = nullptr;
};
struct reset {
  int32_t * error_out = nullptr;
};

}  // namespace emel::tensor::lifetime_analyzer::event

namespace emel::tensor::lifetime_analyzer::events {

struct validate_done {};
struct validate_error {
  int32_t err = 0;
};

struct collect_ranges_done {};
struct collect_ranges_error {
  int32_t err = 0;
};

struct publish_done {};
struct publish_error {
  int32_t err = 0;
};

struct analyze_done {};
struct analyze_error {
  int32_t err = 0;
};

struct reset_done {};
struct reset_error {
  int32_t err = 0;
};

using bootstrap_event = event::analyze;

}  // namespace emel::tensor::lifetime_analyzer::events
