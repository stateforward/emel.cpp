#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/lifetime_analyzer/sm.hpp"

namespace {

TEST_CASE("tensor_lifetime_analyzer_sm_analyze_success_and_reset") {
  emel::tensor::lifetime_analyzer::sm machine{};
  std::array<emel::tensor::lifetime_analyzer::event::tensor_desc, 1> tensors = {{
    emel::tensor::lifetime_analyzer::event::tensor_desc{
      .tensor_id = 1,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};
  std::array<int32_t, 1> first_use = {{-1}};
  std::array<int32_t, 1> last_use = {{-1}};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::tensor::lifetime_analyzer::event::analyze{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .first_use_out = first_use.data(),
    .last_use_out = last_use.data(),
    .ranges_out_count = static_cast<int32_t>(tensors.size()),
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  CHECK(machine.process_event(emel::tensor::lifetime_analyzer::event::reset{
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
}

TEST_CASE("tensor_lifetime_analyzer_sm_validation_error_path") {
  emel::tensor::lifetime_analyzer::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::tensor::lifetime_analyzer::event::analyze{
    .tensors = nullptr,
    .tensor_count = 1,
    .first_use_out = nullptr,
    .last_use_out = nullptr,
    .ranges_out_count = 0,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

}  // namespace
