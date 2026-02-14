#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/lifetime_analyzer/actions.hpp"
#include "emel/tensor/lifetime_analyzer/guards.hpp"
#include "emel/tensor/lifetime_analyzer/sm.hpp"

namespace {

using tensor_desc = emel::tensor::lifetime_analyzer::event::tensor_desc;

TEST_CASE("lifetime_analyzer_starts_idle") {
  emel::tensor::lifetime_analyzer::sm machine{};
  int32_t state_count = 0;
  machine.visit_current_states([&](auto) { state_count += 1; });
  CHECK(state_count == 1);
}

TEST_CASE("lifetime_analyzer_analyze_computes_first_and_last_use") {
  emel::tensor::lifetime_analyzer::sm machine{};

  std::array<tensor_desc, 4> tensors = {{
    tensor_desc{
      .tensor_id = 10,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
    },
    tensor_desc{
      .tensor_id = 11,
      .src_ids = {{10, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
    },
    tensor_desc{
      .tensor_id = 12,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = 10,
    },
    tensor_desc{
      .tensor_id = 13,
      .src_ids = {{12, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
    },
  }};

  std::array<int32_t, 4> first_use = {{-1, -1, -1, -1}};
  std::array<int32_t, 4> last_use = {{-1, -1, -1, -1}};
  int32_t error = -1;

  CHECK(machine.process_event(emel::tensor::lifetime_analyzer::event::analyze{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .first_use_out = first_use.data(),
    .last_use_out = last_use.data(),
    .ranges_out_count = static_cast<int32_t>(first_use.size()),
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(first_use[0] == 0);
  CHECK(first_use[1] == 1);
  CHECK(first_use[2] == 2);
  CHECK(first_use[3] == 3);
  CHECK(last_use[0] == 3);
  CHECK(last_use[1] == 1);
  CHECK(last_use[2] == 3);
  CHECK(last_use[3] == 3);
}

TEST_CASE("lifetime_analyzer_ignores_control_dependency_views_for_n_views") {
  emel::tensor::lifetime_analyzer::sm machine{};

  std::array<tensor_desc, 3> tensors = {{
    tensor_desc{
      .tensor_id = 30,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
    },
    tensor_desc{
      .tensor_id = 31,
      .src_ids = {{30, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
    },
    tensor_desc{
      .tensor_id = 32,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = 30,
      .is_control_dep = true,
    },
  }};

  std::array<int32_t, 3> first_use = {{-1, -1, -1}};
  std::array<int32_t, 3> last_use = {{-1, -1, -1}};
  int32_t error = -1;

  CHECK(machine.process_event(emel::tensor::lifetime_analyzer::event::analyze{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .first_use_out = first_use.data(),
    .last_use_out = last_use.data(),
    .ranges_out_count = static_cast<int32_t>(first_use.size()),
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(last_use[0] == 1);
}

TEST_CASE("lifetime_analyzer_tracks_non_exec_leaf_usage_from_exec_nodes") {
  emel::tensor::lifetime_analyzer::sm machine{};

  std::array<tensor_desc, 3> tensors = {{
    tensor_desc{
      .tensor_id = 40,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = false,
    },
    tensor_desc{
      .tensor_id = 41,
      .src_ids = {{40, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
    },
    tensor_desc{
      .tensor_id = 42,
      .src_ids = {{41, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
    },
  }};

  std::array<int32_t, 3> first_use = {{-1, -1, -1}};
  std::array<int32_t, 3> last_use = {{-1, -1, -1}};
  int32_t error = -1;

  CHECK(machine.process_event(emel::tensor::lifetime_analyzer::event::analyze{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .first_use_out = first_use.data(),
    .last_use_out = last_use.data(),
    .ranges_out_count = static_cast<int32_t>(first_use.size()),
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(last_use[0] == 1);
  CHECK(last_use[1] == 2);
}

TEST_CASE("lifetime_analyzer_reports_invalid_arguments") {
  emel::tensor::lifetime_analyzer::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::tensor::lifetime_analyzer::event::analyze{
    .tensors = nullptr,
    .tensor_count = 1,
    .ranges_out_count = 0,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_action_validate_covers_error_combinations") {
  emel::tensor::lifetime_analyzer::action::context c{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 100,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};

  c.tensors = tensors.data();
  c.tensor_count = 1;
  c.ranges_out_count = 1;
  int32_t first = -1;
  int32_t last = -1;
  c.first_use_out = &first;
  c.last_use_out = &last;

  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_validate(
      emel::tensor::lifetime_analyzer::event::validate{.error_out = &err}, c);
  CHECK(err == EMEL_OK);

  c.tensor_count = -1;
  emel::tensor::lifetime_analyzer::action::run_validate(
      emel::tensor::lifetime_analyzer::event::validate{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  c.tensor_count = 1;
  c.tensors = nullptr;
  emel::tensor::lifetime_analyzer::action::run_validate(
      emel::tensor::lifetime_analyzer::event::validate{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  c.tensors = tensors.data();
  c.ranges_out_count = -1;
  emel::tensor::lifetime_analyzer::action::run_validate(
      emel::tensor::lifetime_analyzer::event::validate{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  c.ranges_out_count = 1;
  c.first_use_out = nullptr;
  emel::tensor::lifetime_analyzer::action::run_validate(
      emel::tensor::lifetime_analyzer::event::validate{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  c.first_use_out = &first;
  c.last_use_out = &last;
  c.ranges_out_count = 0;
  emel::tensor::lifetime_analyzer::action::run_validate(
      emel::tensor::lifetime_analyzer::event::validate{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_action_collect_ranges_error_paths") {
  int32_t err = EMEL_OK;

  std::array<tensor_desc, 1> missing_src = {{
    tensor_desc{
      .tensor_id = 110,
      .src_ids = {{999, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};
  emel::tensor::lifetime_analyzer::action::context c{};
  c.tensors = missing_src.data();
  c.tensor_count = 1;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
      emel::tensor::lifetime_analyzer::event::collect_ranges{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::array<tensor_desc, 2> duplicate_ids = {{
    tensor_desc{
      .tensor_id = 120,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
    tensor_desc{
      .tensor_id = 120,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};
  c = {};
  c.tensors = duplicate_ids.data();
  c.tensor_count = static_cast<int32_t>(duplicate_ids.size());
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
      emel::tensor::lifetime_analyzer::event::collect_ranges{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::array<tensor_desc, 2> non_exec_view_parent = {{
    tensor_desc{
      .tensor_id = 130,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = 131,
      .is_exec_node = false,
      .is_control_dep = false,
    },
    tensor_desc{
      .tensor_id = 132,
      .src_ids = {{130, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};
  c = {};
  c.tensors = non_exec_view_parent.data();
  c.tensor_count = static_cast<int32_t>(non_exec_view_parent.size());
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
      emel::tensor::lifetime_analyzer::event::collect_ranges{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_action_publish_and_outcome_handlers") {
  emel::tensor::lifetime_analyzer::action::context c{};
  c.tensor_count = 2;
  c.first_use[0] = 1;
  c.first_use[1] = 2;
  c.last_use[0] = 3;
  c.last_use[1] = 4;
  int32_t first[2] = {-1, -1};
  int32_t last[2] = {-1, -1};
  c.first_use_out = first;
  c.last_use_out = last;

  int32_t err = -1;
  emel::tensor::lifetime_analyzer::action::run_publish(
      emel::tensor::lifetime_analyzer::event::publish{.error_out = &err}, c);
  CHECK(err == EMEL_OK);
  CHECK(first[0] == 1);
  CHECK(last[1] == 4);

  int32_t boundary_err = -1;
  c.error_out = &boundary_err;
  emel::tensor::lifetime_analyzer::action::on_analyze_done(
      emel::tensor::lifetime_analyzer::events::analyze_done{}, c);
  CHECK(boundary_err == EMEL_OK);

  emel::tensor::lifetime_analyzer::action::on_analyze_error(
      emel::tensor::lifetime_analyzer::events::analyze_error{.err = EMEL_ERR_INVALID_ARGUMENT}, c);
  CHECK(boundary_err == EMEL_ERR_INVALID_ARGUMENT);

  emel::tensor::lifetime_analyzer::action::on_reset_error(
      emel::tensor::lifetime_analyzer::events::reset_error{.err = EMEL_ERR_BACKEND}, c);
  CHECK(boundary_err == EMEL_ERR_BACKEND);

  emel::tensor::lifetime_analyzer::action::record_phase_error(
      emel::tensor::lifetime_analyzer::events::collect_ranges_error{.err = EMEL_ERR_INVALID_ARGUMENT}, c);
  CHECK(boundary_err == EMEL_ERR_INVALID_ARGUMENT);

  emel::tensor::lifetime_analyzer::action::on_reset_done(
      emel::tensor::lifetime_analyzer::events::reset_done{}, c);
  CHECK(c.tensor_count == 0);
}

TEST_CASE("lifetime_analyzer_reports_missing_dependency") {
  emel::tensor::lifetime_analyzer::sm machine{};

  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 11,
      .src_ids = {{99, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
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

TEST_CASE("lifetime_analyzer_reset_clears_runtime_state") {
  emel::tensor::lifetime_analyzer::sm machine{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 21,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
    },
  }};

  CHECK(machine.process_event(emel::tensor::lifetime_analyzer::event::analyze{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .ranges_out_count = 0,
  }));
  CHECK(machine.analyzed_tensor_count() == 1);

  CHECK(machine.process_event(emel::tensor::lifetime_analyzer::event::reset{}));
  CHECK(machine.analyzed_tensor_count() == 0);
}

}  // namespace
