#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/planner/actions.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using graph_view = emel::buffer::allocator::event::graph_view;

struct graph_storage {
  std::array<tensor_desc, 2> nodes = {};
  std::array<tensor_desc, 2> leafs = {};
  int32_t n_nodes = 0;
  int32_t n_leafs = 0;
};

graph_view as_view(const graph_storage & g) {
  return graph_view{
    .nodes = g.nodes.data(),
    .n_nodes = g.n_nodes,
    .leafs = g.leafs.data(),
    .n_leafs = g.n_leafs,
  };
}

graph_storage make_valid_graph() {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 1;
  g.leafs[0] = tensor_desc{
    .tensor_id = 1,
    .alloc_size = 64,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 2,
    .alloc_size = 64,
    .src_ids = {{1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

graph_storage make_duplicate_id_graph() {
  graph_storage g = make_valid_graph();
  g.nodes[0].tensor_id = 1;
  return g;
}

graph_storage make_invalid_view_graph() {
  graph_storage g = make_valid_graph();
  g.nodes[0].is_view = true;
  g.nodes[0].view_src_id = -1;
  return g;
}

graph_storage make_missing_source_graph() {
  graph_storage g = make_valid_graph();
  g.nodes[0].src_ids[0] = 999;
  return g;
}

struct plan_probe {
  int32_t done_calls = 0;
  int32_t error_calls = 0;
  int32_t err = EMEL_OK;
};

bool on_plan_done(void * owner_sm, const emel::buffer::planner::events::plan_done &) {
  auto * owner = static_cast<plan_probe *>(owner_sm);
  if (owner == nullptr) return false;
  owner->done_calls += 1;
  return true;
}

bool on_plan_error(void * owner_sm, const emel::buffer::planner::events::plan_error & ev) {
  auto * owner = static_cast<plan_probe *>(owner_sm);
  if (owner == nullptr) return false;
  owner->error_calls += 1;
  owner->err = ev.err;
  return true;
}

bool run_plan(
    emel::buffer::planner::sm & planner,
    const graph_view & graph,
    const int32_t * node_buffer_ids,
    const int32_t * leaf_buffer_ids,
    const int32_t buffer_count,
    const bool size_only,
    int32_t * sizes_out,
    const int32_t sizes_out_count,
    int32_t * error_out,
    plan_probe & probe) {
  return planner.process_event(emel::buffer::planner::event::plan{
    .graph = graph,
    .node_buffer_ids = node_buffer_ids,
    .leaf_buffer_ids = leaf_buffer_ids,
    .buffer_count = buffer_count,
    .size_only = size_only,
    .sizes_out = sizes_out,
    .sizes_out_count = sizes_out_count,
    .error_out = error_out,
    .owner_sm = &probe,
    .dispatch_done = &on_plan_done,
    .dispatch_error = &on_plan_error,
    .strategy = &emel::buffer::planner::default_strategies::reserve,
  });
}

}  // namespace

TEST_CASE("buffer_planner_valid_plan_event_rejects_invalid_inputs") {
  plan_probe probe{};
  std::array<int32_t, 1> sizes = {{0}};
  const auto g = make_valid_graph();

  emel::buffer::planner::event::plan ev{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = true,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = nullptr,
    .owner_sm = &probe,
    .dispatch_done = &on_plan_done,
    .dispatch_error = &on_plan_error,
    .strategy = &emel::buffer::planner::default_strategies::reserve,
  };

  ev.buffer_count = 0;
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(ev));
}

TEST_CASE("buffer_planner_handles_duplicate_tensor_ids") {
  emel::buffer::planner::sm planner{};
  plan_probe probe{};
  int32_t error = EMEL_OK;
  std::array<int32_t, 1> sizes = {{0}};
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{0}};

  CHECK_FALSE(run_plan(
    planner,
    as_view(make_duplicate_id_graph()),
    node_ids.data(),
    leaf_ids.data(),
    1,
    false,
    sizes.data(),
    static_cast<int32_t>(sizes.size()),
    &error,
    probe));
  CHECK(error != EMEL_OK);
  CHECK(probe.error_calls == 1);
}

TEST_CASE("buffer_planner_handles_invalid_view_source") {
  emel::buffer::planner::sm planner{};
  plan_probe probe{};
  int32_t error = EMEL_OK;
  std::array<int32_t, 1> sizes = {{0}};
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{0}};

  CHECK_FALSE(run_plan(
    planner,
    as_view(make_invalid_view_graph()),
    node_ids.data(),
    leaf_ids.data(),
    1,
    false,
    sizes.data(),
    static_cast<int32_t>(sizes.size()),
    &error,
    probe));
  CHECK(error != EMEL_OK);
  CHECK(probe.error_calls == 1);
}

TEST_CASE("buffer_planner_handles_missing_sources") {
  emel::buffer::planner::sm planner{};
  plan_probe probe{};
  int32_t error = EMEL_OK;
  std::array<int32_t, 1> sizes = {{0}};
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{0}};

  CHECK_FALSE(run_plan(
    planner,
    as_view(make_missing_source_graph()),
    node_ids.data(),
    leaf_ids.data(),
    1,
    false,
    sizes.data(),
    static_cast<int32_t>(sizes.size()),
    &error,
    probe));
  CHECK(error != EMEL_OK);
  CHECK(probe.error_calls == 1);
}

TEST_CASE("buffer_planner_handles_invalid_buffer_mapping") {
  emel::buffer::planner::sm planner{};
  plan_probe probe{};
  int32_t error = EMEL_OK;
  std::array<int32_t, 1> sizes = {{0}};
  std::array<int32_t, 1> node_ids = {{2}};
  std::array<int32_t, 1> leaf_ids = {{0}};

  CHECK_FALSE(run_plan(
    planner,
    as_view(make_valid_graph()),
    node_ids.data(),
    leaf_ids.data(),
    1,
    false,
    sizes.data(),
    static_cast<int32_t>(sizes.size()),
    &error,
    probe));
  CHECK(error != EMEL_OK);
  CHECK(probe.error_calls == 1);
}

TEST_CASE("buffer_planner_handles_invalid_sizes_out_configuration") {
  emel::buffer::planner::sm planner{};
  plan_probe probe{};
  int32_t error = EMEL_OK;
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{0}};

  CHECK(run_plan(
    planner,
    as_view(make_valid_graph()),
    node_ids.data(),
    leaf_ids.data(),
    1,
    false,
    nullptr,
    1,
    &error,
    probe));
  CHECK(error == EMEL_OK);
  CHECK(probe.error_calls == 0);
}

TEST_CASE("buffer_planner_successful_plan_reports_sizes") {
  emel::buffer::planner::sm planner{};
  plan_probe probe{};
  int32_t error = EMEL_OK;
  std::array<int32_t, 1> sizes = {{0}};
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{0}};

  CHECK(run_plan(
    planner,
    as_view(make_valid_graph()),
    node_ids.data(),
    leaf_ids.data(),
    1,
    true,
    sizes.data(),
    static_cast<int32_t>(sizes.size()),
    &error,
    probe));
  CHECK(error == EMEL_OK);
  CHECK(probe.done_calls == 1);
  CHECK(sizes[0] >= 0);
}
