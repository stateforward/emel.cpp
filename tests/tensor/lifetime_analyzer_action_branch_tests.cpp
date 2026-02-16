#include <array>
#include <doctest/doctest.h>

#include "emel/tensor/lifetime_analyzer/actions.hpp"
#include "emel/tensor/lifetime_analyzer/events.hpp"
#include "emel/emel.h"

TEST_CASE("lifetime_analyzer_run_validate_rejects_invalid_inputs") {
  emel::tensor::lifetime_analyzer::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::tensor::lifetime_analyzer::action::run_validate(
    emel::tensor::lifetime_analyzer::event::validate{
      .tensors = nullptr,
      .tensor_count = -1,
      .ranges_out_count = 0,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_validate(
    emel::tensor::lifetime_analyzer::event::validate{
      .tensors = nullptr,
      .tensor_count = 1,
      .ranges_out_count = 0,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_run_collect_ranges_detects_invalid_ids") {
  emel::tensor::lifetime_analyzer::action::context ctx{};
  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::event::tensor_desc tensor{};

  tensor.tensor_id = -1;
  ctx.tensor_count = 1;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = &tensor,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_run_validate_rejects_missing_outputs") {
  emel::tensor::lifetime_analyzer::action::context ctx{};
  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::event::tensor_desc tensor{};
  tensor.tensor_id = 1;

  emel::tensor::lifetime_analyzer::action::run_validate(
    emel::tensor::lifetime_analyzer::event::validate{
      .tensors = &tensor,
      .tensor_count = 1,
      .first_use_out = nullptr,
      .last_use_out = nullptr,
      .ranges_out_count = 1,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::array<int32_t, 1> first_use = {{-1}};
  std::array<int32_t, 1> last_use = {{-1}};
  err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_validate(
    emel::tensor::lifetime_analyzer::event::validate{
      .tensors = &tensor,
      .tensor_count = 2,
      .first_use_out = first_use.data(),
      .last_use_out = last_use.data(),
      .ranges_out_count = 1,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_run_validate_accepts_valid_inputs") {
  emel::tensor::lifetime_analyzer::action::context ctx{};
  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::event::tensor_desc tensor{};
  tensor.tensor_id = 1;
  std::array<int32_t, 1> first_use = {{-1}};
  std::array<int32_t, 1> last_use = {{-1}};

  emel::tensor::lifetime_analyzer::action::run_validate(
    emel::tensor::lifetime_analyzer::event::validate{
      .tensors = &tensor,
      .tensor_count = 1,
      .first_use_out = first_use.data(),
      .last_use_out = last_use.data(),
      .ranges_out_count = 1,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.tensor_count == 1);
}

TEST_CASE("lifetime_analyzer_detail_helpers_cover_branches") {
  using emel::tensor::lifetime_analyzer::action::detail::has_duplicate_ids;
  using emel::tensor::lifetime_analyzer::action::detail::index_of_tensor;
  using emel::tensor::lifetime_analyzer::action::detail::normalize_error;

  emel::tensor::lifetime_analyzer::action::context ctx{};
  ctx.tensor_count = 2;
  ctx.tensor_ids[0] = 3;
  ctx.tensor_ids[1] = 7;

  CHECK(index_of_tensor(ctx, 7) == 1);
  CHECK(index_of_tensor(ctx, 11) == -1);
  CHECK_FALSE(has_duplicate_ids(ctx, 1));

  ctx.tensor_ids[1] = 3;
  CHECK(has_duplicate_ids(ctx, 1));

  CHECK(normalize_error(EMEL_ERR_INVALID_ARGUMENT, EMEL_OK) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(normalize_error(EMEL_OK, EMEL_ERR_INVALID_ARGUMENT) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(normalize_error(EMEL_OK, EMEL_OK) == EMEL_ERR_BACKEND);
}

TEST_CASE("lifetime_analyzer_run_collect_ranges_validates_views_and_sources") {
  emel::tensor::lifetime_analyzer::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::tensor::lifetime_analyzer::event::tensor_desc bad_view{};
  bad_view.tensor_id = 1;
  bad_view.is_view = true;
  bad_view.view_src_id = -1;
  bad_view.is_exec_node = true;

  ctx.tensor_count = 1;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = &bad_view,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::array<emel::tensor::lifetime_analyzer::event::tensor_desc, 2> tensors = {{
    emel::tensor::lifetime_analyzer::event::tensor_desc{
      .tensor_id = 1,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
    emel::tensor::lifetime_analyzer::event::tensor_desc{
      .tensor_id = 2,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = 99,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};
  ctx.tensor_count = static_cast<int32_t>(tensors.size());
  err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = tensors.data(),
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  tensors[1].is_view = false;
  tensors[1].src_ids[0] = 99;
  err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = tensors.data(),
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("lifetime_analyzer_run_collect_ranges_view_chain_updates_last_use") {
  using tensor_desc = emel::tensor::lifetime_analyzer::event::tensor_desc;

  std::array<tensor_desc, 3> tensors = {{
    tensor_desc{
      .tensor_id = 1,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
    tensor_desc{
      .tensor_id = 2,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = 1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
    tensor_desc{
      .tensor_id = 3,
      .src_ids = {{2, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_exec_node = true,
      .is_control_dep = false,
    },
  }};

  emel::tensor::lifetime_analyzer::action::context ctx{};
  ctx.tensor_count = static_cast<int32_t>(tensors.size());
  int32_t err = EMEL_OK;

  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = tensors.data(),
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.first_use[0] == 0);
  CHECK(ctx.last_use[0] == 2);
  CHECK(ctx.first_use[1] == 1);
  CHECK(ctx.last_use[1] == 2);
  CHECK(ctx.first_use[2] == 2);
  CHECK(ctx.last_use[2] == 2);

  std::array<int32_t, 3> first_use = {{-1, -1, -1}};
  std::array<int32_t, 3> last_use = {{-1, -1, -1}};
  err = EMEL_OK;
  emel::tensor::lifetime_analyzer::action::run_publish(
    emel::tensor::lifetime_analyzer::event::publish{
      .first_use_out = first_use.data(),
      .last_use_out = last_use.data(),
      .ranges_out_count = static_cast<int32_t>(first_use.size()),
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(first_use[0] == 0);
  CHECK(last_use[0] == 2);
}

TEST_CASE("lifetime_analyzer_run_collect_ranges_skips_non_exec_nodes") {
  emel::tensor::lifetime_analyzer::action::context ctx{};
  int32_t err = EMEL_OK;
  emel::tensor::lifetime_analyzer::event::tensor_desc tensor{};
  tensor.tensor_id = 4;
  tensor.is_exec_node = false;

  ctx.tensor_count = 1;
  emel::tensor::lifetime_analyzer::action::run_collect_ranges(
    emel::tensor::lifetime_analyzer::event::collect_ranges{
      .tensors = &tensor,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.first_use[0] == 0);
  CHECK(ctx.last_use[0] == 0);
}
