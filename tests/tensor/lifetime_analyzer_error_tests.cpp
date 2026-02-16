#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/lifetime_analyzer/actions.hpp"
#include "emel/tensor/lifetime_analyzer/sm.hpp"

namespace {

using tensor_desc = emel::tensor::lifetime_analyzer::event::tensor_desc;

}  // namespace

TEST_CASE("lifetime_analyzer_collect_ranges_reports_duplicate_ids") {
  emel::tensor::lifetime_analyzer::action::context c{};
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

  emel::tensor::lifetime_analyzer::action::begin_analyze(
    emel::tensor::lifetime_analyzer::event::analyze{
      .tensor_count = static_cast<int32_t>(tensors.size()),
    },
    c);

  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = tensors.data(),
      .tensor_count = static_cast<int32_t>(tensors.size()),
      .error_out = &err,
    },
    c);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_collect_ranges_reports_missing_sources") {
  emel::tensor::lifetime_analyzer::action::context c{};
  std::array<tensor_desc, 2> tensors = {{
    tensor_desc{
      .tensor_id = 10,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
    tensor_desc{
      .tensor_id = 11,
      .src_ids = {{999, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};

  emel::tensor::lifetime_analyzer::action::begin_analyze(
    emel::tensor::lifetime_analyzer::event::analyze{
      .tensor_count = static_cast<int32_t>(tensors.size()),
    },
    c);

  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = tensors.data(),
      .tensor_count = static_cast<int32_t>(tensors.size()),
      .error_out = &err,
    },
    c);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_collect_ranges_reports_missing_view_source") {
  emel::tensor::lifetime_analyzer::action::context c{};
  std::array<tensor_desc, 2> tensors = {{
    tensor_desc{
      .tensor_id = 20,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
    tensor_desc{
      .tensor_id = 21,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = 999,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};

  emel::tensor::lifetime_analyzer::action::begin_analyze(
    emel::tensor::lifetime_analyzer::event::analyze{
      .tensor_count = static_cast<int32_t>(tensors.size()),
    },
    c);

  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = tensors.data(),
      .tensor_count = static_cast<int32_t>(tensors.size()),
      .error_out = &err,
    },
    c);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_validate_rejects_ranges_out_count_mismatch") {
  emel::tensor::lifetime_analyzer::action::context c{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 30,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};

  int32_t first = -1;
  int32_t last = -1;
  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_validate(
    emel::tensor::lifetime_analyzer::event::validate{
      .tensors = tensors.data(),
      .tensor_count = static_cast<int32_t>(tensors.size()),
      .first_use_out = &first,
      .last_use_out = &last,
      .ranges_out_count = 0,
      .error_out = &err,
    },
    c);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_reset_completes_successfully") {
  emel::tensor::lifetime_analyzer::sm machine{};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::tensor::lifetime_analyzer::event::reset{
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
}

TEST_CASE("lifetime_analyzer_collect_ranges_rejects_negative_tensor_id") {
  emel::tensor::lifetime_analyzer::action::context c{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = -1,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};

  emel::tensor::lifetime_analyzer::action::begin_analyze(
    emel::tensor::lifetime_analyzer::event::analyze{
      .tensor_count = static_cast<int32_t>(tensors.size()),
    },
    c);

  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = tensors.data(),
      .tensor_count = static_cast<int32_t>(tensors.size()),
      .error_out = &err,
    },
    c);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_collect_ranges_rejects_view_without_source_id") {
  emel::tensor::lifetime_analyzer::action::context c{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 101,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};

  emel::tensor::lifetime_analyzer::action::begin_analyze(
    emel::tensor::lifetime_analyzer::event::analyze{
      .tensor_count = static_cast<int32_t>(tensors.size()),
    },
    c);

  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = tensors.data(),
      .tensor_count = static_cast<int32_t>(tensors.size()),
      .error_out = &err,
    },
    c);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_collect_ranges_rejects_missing_parent_index") {
  emel::tensor::lifetime_analyzer::action::context c{};
  std::array<tensor_desc, 2> tensors = {{
    tensor_desc{
      .tensor_id = 201,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
    tensor_desc{
      .tensor_id = 202,
      .src_ids = {{999, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};

  emel::tensor::lifetime_analyzer::action::begin_analyze(
    emel::tensor::lifetime_analyzer::event::analyze{
      .tensor_count = static_cast<int32_t>(tensors.size()),
    },
    c);

  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = tensors.data(),
      .tensor_count = static_cast<int32_t>(tensors.size()),
      .error_out = &err,
    },
    c);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}
