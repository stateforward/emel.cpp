#include <array>
#include <cstdint>
#include <limits>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/realloc_analyzer/actions.hpp"
#include "emel/buffer/realloc_analyzer/guards.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using graph_view = emel::buffer::realloc_analyzer::event::graph_view;
using node_alloc = emel::buffer::realloc_analyzer::event::node_alloc;
using leaf_alloc = emel::buffer::realloc_analyzer::event::leaf_alloc;

TEST_CASE("buffer_realloc_analyzer_starts_idle") {
  emel::buffer::realloc_analyzer::sm machine{};
  int32_t state_count = 0;
  machine.visit_current_states([&](auto) { state_count += 1; });
  CHECK(state_count == 1);
}

TEST_CASE("buffer_realloc_analyzer_reports_no_realloc_for_valid_snapshot") {
  emel::buffer::realloc_analyzer::sm machine{};

  std::array<tensor_desc, 1> nodes = {{
    tensor_desc{
      .tensor_id = 2,
      .alloc_size = 64,
      .src_ids = {{1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = false,
      .is_output = false,
      .has_external_data = false,
    },
  }};
  std::array<tensor_desc, 1> leafs = {{
    tensor_desc{
      .tensor_id = 1,
      .alloc_size = 32,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = true,
      .is_output = false,
      .has_external_data = false,
    },
  }};
  std::array<node_alloc, 1> node_allocs = {{
    node_alloc{
      .dst = {.buffer_id = 0, .size_max = 64},
      .src = {{{.buffer_id = 0, .size_max = 32}, {}, {}, {}}},
    },
  }};
  std::array<leaf_alloc, 1> leaf_allocs = {{
    leaf_alloc{.leaf = {.buffer_id = 0, .size_max = 32}},
  }};

  int32_t needs_realloc = -1;
  int32_t error = -1;
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = graph_view{
      .nodes = nodes.data(),
      .n_nodes = static_cast<int32_t>(nodes.size()),
      .leafs = leafs.data(),
      .n_leafs = static_cast<int32_t>(leafs.size()),
    },
    .node_allocs = node_allocs.data(),
    .node_alloc_count = static_cast<int32_t>(node_allocs.size()),
    .leaf_allocs = leaf_allocs.data(),
    .leaf_alloc_count = static_cast<int32_t>(leaf_allocs.size()),
    .needs_realloc_out = &needs_realloc,
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(needs_realloc == 0);
  CHECK_FALSE(machine.needs_realloc());
}

TEST_CASE("buffer_realloc_analyzer_requires_realloc_on_snapshot_shape_mismatch") {
  emel::buffer::realloc_analyzer::sm machine{};

  std::array<tensor_desc, 1> nodes = {{
    tensor_desc{
      .tensor_id = 10,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = false,
      .is_output = false,
      .has_external_data = false,
    },
  }};
  std::array<node_alloc, 1> node_allocs = {{
    node_alloc{
      .dst = {.buffer_id = 0, .size_max = 16},
      .src = {{{}, {}, {}, {}}},
    },
  }};

  int32_t needs_realloc = -1;
  int32_t error = -1;
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = graph_view{
      .nodes = nodes.data(),
      .n_nodes = static_cast<int32_t>(nodes.size()),
      .leafs = nullptr,
      .n_leafs = 0,
    },
    .node_allocs = node_allocs.data(),
    .node_alloc_count = 0,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = &needs_realloc,
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(needs_realloc == 1);
}

TEST_CASE("buffer_realloc_analyzer_reports_invalid_payload") {
  emel::buffer::realloc_analyzer::sm machine{};

  int32_t needs_realloc = -1;
  int32_t error = EMEL_OK;
  CHECK_FALSE(machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = graph_view{
      .nodes = nullptr,
      .n_nodes = 1,
      .leafs = nullptr,
      .n_leafs = 0,
    },
    .node_allocs = nullptr,
    .node_alloc_count = 0,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = &needs_realloc,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_realloc_analyzer_requires_realloc_for_missing_src_or_small_snapshot") {
  emel::buffer::realloc_analyzer::sm machine{};

  std::array<tensor_desc, 1> nodes = {{
    tensor_desc{
      .tensor_id = 20,
      .alloc_size = 64,
      .src_ids = {{999, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = false,
      .is_output = false,
      .has_external_data = false,
    },
  }};
  std::array<node_alloc, 1> node_allocs = {{
    node_alloc{
      .dst = {.buffer_id = 0, .size_max = 32},
      .src = {{{.buffer_id = 0, .size_max = 8}, {}, {}, {}}},
    },
  }};

  int32_t needs_realloc = -1;
  int32_t error = -1;
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = graph_view{
      .nodes = nodes.data(),
      .n_nodes = static_cast<int32_t>(nodes.size()),
      .leafs = nullptr,
      .n_leafs = 0,
    },
    .node_allocs = node_allocs.data(),
    .node_alloc_count = static_cast<int32_t>(node_allocs.size()),
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = &needs_realloc,
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(needs_realloc == 1);
}

TEST_CASE("buffer_realloc_analyzer_treats_external_or_view_as_no_size_requirement") {
  emel::buffer::realloc_analyzer::sm machine{};

  std::array<tensor_desc, 1> nodes = {{
    tensor_desc{
      .tensor_id = 30,
      .alloc_size = 4096,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = false,
      .is_output = false,
      .has_external_data = true,
    },
  }};
  std::array<node_alloc, 1> node_allocs = {{
    node_alloc{
      .dst = {.buffer_id = -1, .size_max = 0},
      .src = {{{}, {}, {}, {}}},
    },
  }};

  int32_t needs_realloc = -1;
  int32_t error = -1;
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = graph_view{
      .nodes = nodes.data(),
      .n_nodes = static_cast<int32_t>(nodes.size()),
      .leafs = nullptr,
      .n_leafs = 0,
    },
    .node_allocs = node_allocs.data(),
    .node_alloc_count = static_cast<int32_t>(node_allocs.size()),
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = &needs_realloc,
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(needs_realloc == 0);
}

TEST_CASE("buffer_realloc_analyzer_reset_and_guards") {
  emel::buffer::realloc_analyzer::sm machine{};

  std::array<tensor_desc, 1> nodes = {{
    tensor_desc{
      .tensor_id = 40,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = false,
      .is_output = false,
      .has_external_data = false,
    },
  }};
  std::array<node_alloc, 1> node_allocs = {{
    node_alloc{
      .dst = {.buffer_id = -1, .size_max = 0},
      .src = {{{}, {}, {}, {}}},
    },
  }};
  int32_t needs_realloc = -1;
  int32_t error = -1;

  CHECK(machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = graph_view{
      .nodes = nodes.data(),
      .n_nodes = static_cast<int32_t>(nodes.size()),
      .leafs = nullptr,
      .n_leafs = 0,
    },
    .node_allocs = node_allocs.data(),
    .node_alloc_count = static_cast<int32_t>(node_allocs.size()),
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = &needs_realloc,
    .error_out = &error,
  }));
  CHECK(needs_realloc == 1);
  CHECK(machine.needs_realloc());

  CHECK(machine.process_event(emel::buffer::realloc_analyzer::event::reset{}));
  CHECK_FALSE(machine.needs_realloc());

  emel::buffer::realloc_analyzer::action::context ctx{};
  CHECK(
      emel::buffer::realloc_analyzer::guard::no_error{}(
          emel::buffer::realloc_analyzer::events::validate_error{.err = EMEL_OK}, ctx));
  CHECK(
      emel::buffer::realloc_analyzer::guard::has_error{}(
          emel::buffer::realloc_analyzer::events::validate_error{.err = EMEL_ERR_BACKEND}, ctx));
}

TEST_CASE("buffer_realloc_analyzer_action_detail_helpers_cover_fallbacks") {
  namespace action = emel::buffer::realloc_analyzer::action;
  CHECK(action::detail::normalize_error(EMEL_ERR_INVALID_ARGUMENT, EMEL_ERR_BACKEND) ==
        EMEL_ERR_INVALID_ARGUMENT);
  CHECK(action::detail::normalize_error(EMEL_OK, EMEL_ERR_INVALID_ARGUMENT) ==
        EMEL_ERR_INVALID_ARGUMENT);
  CHECK(action::detail::normalize_error(EMEL_OK, EMEL_OK) == EMEL_ERR_BACKEND);

  CHECK(action::detail::align_up_16(0) == 0);
  CHECK(action::detail::align_up_16(std::numeric_limits<int32_t>::max()) ==
        std::numeric_limits<int32_t>::max());

  std::array<tensor_desc, 1> nodes = {{
    tensor_desc{
      .tensor_id = 50,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = false,
      .is_output = false,
      .has_external_data = false,
    },
  }};
  std::array<tensor_desc, 1> leafs = {{
    tensor_desc{
      .tensor_id = 51,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = true,
      .is_output = false,
      .has_external_data = false,
    },
  }};
  const auto graph = graph_view{
    .nodes = nodes.data(),
    .n_nodes = static_cast<int32_t>(nodes.size()),
    .leafs = leafs.data(),
    .n_leafs = static_cast<int32_t>(leafs.size()),
  };
  CHECK(action::detail::find_tensor(graph, 50) != nullptr);
  CHECK(action::detail::find_tensor(graph, 51) != nullptr);
  CHECK(action::detail::find_tensor(graph, 999) == nullptr);
}

TEST_CASE("buffer_realloc_analyzer_action_validate_leaf_alloc_pointer_branch") {
  namespace action = emel::buffer::realloc_analyzer::action;
  action::context c{};
  std::array<tensor_desc, 1> leafs = {{
    tensor_desc{
      .tensor_id = 60,
      .alloc_size = 32,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = true,
      .is_output = false,
      .has_external_data = false,
    },
  }};
  c.graph = graph_view{
    .nodes = nullptr,
    .n_nodes = 0,
    .leafs = leafs.data(),
    .n_leafs = static_cast<int32_t>(leafs.size()),
  };
  c.node_alloc_count = 0;
  c.leaf_alloc_count = static_cast<int32_t>(leafs.size());
  c.leaf_allocs = nullptr;

  int32_t err = EMEL_OK;
  action::run_validate(emel::buffer::realloc_analyzer::event::validate{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_realloc_analyzer_action_evaluate_handles_leaf_mismatch_and_src_snapshot_size") {
  namespace action = emel::buffer::realloc_analyzer::action;

  {
    action::context c{};
    std::array<tensor_desc, 1> leafs = {{
      tensor_desc{
        .tensor_id = 70,
        .alloc_size = 16,
        .src_ids = {{-1, -1, -1, -1}},
        .is_view = false,
        .view_src_id = -1,
        .is_input = true,
        .is_output = false,
        .has_external_data = false,
      },
    }};
    c.graph = graph_view{
      .nodes = nullptr,
      .n_nodes = 0,
      .leafs = leafs.data(),
      .n_leafs = static_cast<int32_t>(leafs.size()),
    };
    c.node_alloc_count = 0;
    c.leaf_alloc_count = 0;
    action::run_evaluate(emel::buffer::realloc_analyzer::event::evaluate{}, c);
    CHECK(c.needs_realloc);
  }

  {
    action::context c{};
    std::array<tensor_desc, 1> nodes = {{
      tensor_desc{
        .tensor_id = 71,
        .alloc_size = 64,
        .src_ids = {{72, -1, -1, -1}},
        .is_view = false,
        .view_src_id = -1,
        .is_input = false,
        .is_output = false,
        .has_external_data = false,
      },
    }};
    std::array<tensor_desc, 1> leafs = {{
      tensor_desc{
        .tensor_id = 72,
        .alloc_size = 32,
        .src_ids = {{-1, -1, -1, -1}},
        .is_view = false,
        .view_src_id = -1,
        .is_input = true,
        .is_output = false,
        .has_external_data = false,
      },
    }};
    std::array<node_alloc, 1> node_allocs = {{
      node_alloc{
        .dst = {.buffer_id = 0, .size_max = 64},
        .src = {{{.buffer_id = 0, .size_max = 8}, {}, {}, {}}},
      },
    }};
    std::array<leaf_alloc, 1> leaf_allocs = {{
      leaf_alloc{.leaf = {.buffer_id = 0, .size_max = 32}},
    }};
    c.graph = graph_view{
      .nodes = nodes.data(),
      .n_nodes = static_cast<int32_t>(nodes.size()),
      .leafs = leafs.data(),
      .n_leafs = static_cast<int32_t>(leafs.size()),
    };
    c.node_allocs = node_allocs.data();
    c.node_alloc_count = static_cast<int32_t>(node_allocs.size());
    c.leaf_allocs = leaf_allocs.data();
    c.leaf_alloc_count = static_cast<int32_t>(leaf_allocs.size());
    int32_t err = EMEL_ERR_BACKEND;
    action::run_evaluate(emel::buffer::realloc_analyzer::event::evaluate{.error_out = &err}, c);
    CHECK(err == EMEL_OK);
    CHECK(c.needs_realloc);
  }
}

TEST_CASE("buffer_realloc_analyzer_error_handlers_cover_with_and_without_output") {
  namespace action = emel::buffer::realloc_analyzer::action;

  action::context c{};
  int32_t err = EMEL_OK;
  c.error_out = &err;
  action::on_analyze_error(emel::buffer::realloc_analyzer::events::analyze_error{.err = EMEL_OK}, c);
  CHECK(err == EMEL_ERR_BACKEND);
  action::on_reset_error(emel::buffer::realloc_analyzer::events::reset_error{.err = EMEL_OK}, c);
  CHECK(err == EMEL_ERR_BACKEND);

  c.error_out = nullptr;
  const auto step_before = c.step;
  action::on_analyze_error(
      emel::buffer::realloc_analyzer::events::analyze_error{.err = EMEL_ERR_INVALID_ARGUMENT}, c);
  action::on_reset_error(
      emel::buffer::realloc_analyzer::events::reset_error{.err = EMEL_ERR_INVALID_ARGUMENT}, c);
  CHECK(c.step == step_before + 2);
}

}  // namespace
