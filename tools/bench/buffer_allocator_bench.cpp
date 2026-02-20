#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstdio>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/allocator/sm.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using graph_view = emel::buffer::allocator::event::graph_view;

struct graph_storage {
  std::array<tensor_desc, 1> nodes = {};
  std::array<tensor_desc, 1> leafs = {};
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

graph_storage make_graph() {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 1;
  g.leafs[0] = tensor_desc{
    .tensor_id = 10,
    .alloc_size = 256,
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  auto src_ids = emel::buffer::allocator::event::make_src_ids();
  src_ids[0] = 10;
  g.nodes[0] = tensor_desc{
    .tensor_id = 20,
    .alloc_size = 256,
    .src_ids = src_ids,
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

struct reference_state {
  ggml_context * ctx = nullptr;
  ggml_cgraph * graph = nullptr;
  ggml_gallocr_t galloc = nullptr;
};

reference_state make_reference_state() {
  ggml_init_params params = {
    .mem_size = 4 * 1024 * 1024,
    .mem_buffer = nullptr,
    .no_alloc = true,
  };

  ggml_context * ctx = ggml_init(params);
  if (ctx == nullptr) {
    std::fprintf(stderr, "error: ggml_init failed\n");
    std::abort();
  }

  ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
  if (input == nullptr) {
    std::fprintf(stderr, "error: tensor init failed\n");
    ggml_free(ctx);
    std::abort();
  }
  ggml_set_input(input);

  ggml_tensor * node = ggml_dup(ctx, input);
  if (node == nullptr) {
    std::fprintf(stderr, "error: node init failed\n");
    ggml_free(ctx);
    std::abort();
  }
  ggml_set_output(node);

  ggml_cgraph * graph = ggml_new_graph(ctx);
  if (graph == nullptr) {
    std::fprintf(stderr, "error: graph init failed\n");
    ggml_free(ctx);
    std::abort();
  }
  ggml_build_forward_expand(graph, node);

  ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
  if (galloc == nullptr) {
    std::fprintf(stderr, "error: ggml_gallocr_new failed\n");
    ggml_free(ctx);
    std::abort();
  }

  return reference_state{
    .ctx = ctx,
    .graph = graph,
    .galloc = galloc,
  };
}

void free_reference_state(reference_state & state) {
  if (state.galloc != nullptr) {
    ggml_gallocr_free(state.galloc);
    state.galloc = nullptr;
  }
  if (state.ctx != nullptr) {
    ggml_free(state.ctx);
    state.ctx = nullptr;
  }
  state.graph = nullptr;
}

}  // namespace

namespace emel::bench {

void append_emel_buffer_allocator_cases(std::vector<result> & results, const config & cfg) {
  {
    emel::buffer::allocator::sm machine{};
    graph_storage g = make_graph();
    const std::array<int32_t, 1> node_ids = {{0}};
    const std::array<int32_t, 1> leaf_ids = {{0}};

    (void)machine.process_event(emel::buffer::allocator::event::initialize{
      .buffer_count = 1,
    });

    auto fn = [&]() {
      (void)machine.process_event(emel::buffer::allocator::event::reserve_n{
        .graph = as_view(g),
        .node_buffer_ids = node_ids.data(),
        .leaf_buffer_ids = leaf_ids.data(),
      });
    };

    results.push_back(measure_case("buffer/allocator_reserve_n", cfg, fn));
    (void)machine.process_event(emel::buffer::allocator::event::release{});
  }

  {
    emel::buffer::allocator::sm machine{};
    graph_storage g = make_graph();
    const std::array<int32_t, 1> node_ids = {{0}};
    const std::array<int32_t, 1> leaf_ids = {{0}};

    (void)machine.process_event(emel::buffer::allocator::event::initialize{
      .buffer_count = 1,
    });
    (void)machine.process_event(emel::buffer::allocator::event::reserve_n{
      .graph = as_view(g),
      .node_buffer_ids = node_ids.data(),
      .leaf_buffer_ids = leaf_ids.data(),
    });

    auto fn = [&]() {
      (void)machine.process_event(emel::buffer::allocator::event::alloc_graph{
        .graph = as_view(g),
      });
    };

    results.push_back(measure_case("buffer/allocator_alloc_graph", cfg, fn));
    (void)machine.process_event(emel::buffer::allocator::event::release{});
  }

  {
    emel::buffer::allocator::sm machine{};
    graph_storage g = make_graph();
    const std::array<int32_t, 1> node_ids = {{0}};
    const std::array<int32_t, 1> leaf_ids = {{0}};

    (void)machine.process_event(emel::buffer::allocator::event::initialize{
      .buffer_count = 1,
    });

    auto fn = [&]() {
      (void)machine.process_event(emel::buffer::allocator::event::reserve_n{
        .graph = as_view(g),
        .node_buffer_ids = node_ids.data(),
        .leaf_buffer_ids = leaf_ids.data(),
      });
      (void)machine.process_event(emel::buffer::allocator::event::alloc_graph{
        .graph = as_view(g),
      });
    };

    results.push_back(measure_case("buffer/allocator_full", cfg, fn));
    (void)machine.process_event(emel::buffer::allocator::event::release{});
  }
}

void append_reference_buffer_allocator_cases(std::vector<result> & results, const config & cfg) {
  const int node_buffer_ids[1] = {0};
  const int leaf_buffer_ids[1] = {0};

  {
    reference_state state = make_reference_state();
    auto fn = [&]() {
      (void)ggml_gallocr_reserve_n(state.galloc, state.graph, node_buffer_ids, leaf_buffer_ids);
    };
    results.push_back(measure_case("buffer/allocator_reserve_n", cfg, fn));
    free_reference_state(state);
  }

  {
    reference_state state = make_reference_state();
    auto fn = [&]() {
      (void)ggml_gallocr_alloc_graph(state.galloc, state.graph);
    };
    results.push_back(measure_case("buffer/allocator_alloc_graph", cfg, fn));
    free_reference_state(state);
  }

  {
    reference_state state = make_reference_state();
    auto fn = [&]() {
      (void)ggml_gallocr_reserve_n(state.galloc, state.graph, node_buffer_ids, leaf_buffer_ids);
      (void)ggml_gallocr_alloc_graph(state.galloc, state.graph);
    };
    results.push_back(measure_case("buffer/allocator_full", cfg, fn));
    free_reference_state(state);
  }
}

}  // namespace emel::bench
