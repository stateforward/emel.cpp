#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/lifetime_analyzer/sm.hpp"

namespace {

using tensor_desc = emel::tensor::lifetime_analyzer::event::tensor_desc;

}  // namespace

TEST_CASE("lifetime_analyzer_analyze_reports_collect_ranges_error") {
  emel::tensor::lifetime_analyzer::sm machine{};
  std::array<tensor_desc, 2> tensors = {{
    tensor_desc{
      .tensor_id = 1,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
    tensor_desc{
      .tensor_id = 1,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};

  int32_t error = EMEL_OK;
  CHECK_FALSE(machine.process_event(emel::tensor::lifetime_analyzer::event::analyze{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .ranges_out_count = 0,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}
