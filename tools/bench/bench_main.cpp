#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/allocator/sm.hpp"
#include "emel/emel.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace {

constexpr std::uint64_t k_default_iterations = 100000;
constexpr std::size_t k_default_runs = 5;
constexpr std::uint64_t k_default_warmup_iterations = 1000;
constexpr std::size_t k_default_warmup_runs = 1;
constexpr std::size_t k_max_runs = 25;

struct config {
  std::uint64_t iterations = k_default_iterations;
  std::size_t runs = k_default_runs;
  std::uint64_t warmup_iterations = k_default_warmup_iterations;
  std::size_t warmup_runs = k_default_warmup_runs;
};

struct result {
  std::string name;
  double ns_per_op = 0.0;
  std::uint64_t iterations = 0;
  std::size_t runs = 0;
};

std::uint64_t read_env_u64(const char * name, std::uint64_t fallback) {
  const char * value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char * end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value) {
    return fallback;
  }
  return static_cast<std::uint64_t>(parsed);
}

std::size_t read_env_size(const char * name, std::size_t fallback) {
  const auto parsed = read_env_u64(name, static_cast<std::uint64_t>(fallback));
  if (parsed == 0) {
    return fallback;
  }
  if (parsed > static_cast<std::uint64_t>(k_max_runs)) {
    return k_max_runs;
  }
  return static_cast<std::size_t>(parsed);
}

template <class Fn>
result measure_case(const char * name, const config & cfg, Fn && fn) {
  std::vector<double> samples;
  samples.reserve(cfg.runs);

  for (std::size_t run = 0; run < cfg.warmup_runs; ++run) {
    for (std::uint64_t i = 0; i < cfg.warmup_iterations; ++i) {
      fn();
    }
  }

  for (std::size_t run = 0; run < cfg.runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < cfg.iterations; ++i) {
      fn();
    }
    const auto end = std::chrono::steady_clock::now();
    const auto duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    samples.push_back(static_cast<double>(duration_ns) / static_cast<double>(cfg.iterations));
  }

  std::sort(samples.begin(), samples.end());
  const double median = samples[samples.size() / 2];

  result out;
  out.name = name;
  out.ns_per_op = median;
  out.iterations = cfg.iterations;
  out.runs = cfg.runs;
  return out;
}

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

std::vector<result> run_emel_benchmarks(const config & cfg) {
  std::vector<result> results;
  results.reserve(3);

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

  return results;
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

std::vector<result> run_reference_benchmarks(const config & cfg) {
  std::vector<result> results;
  results.reserve(3);

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

  return results;
}

void print_snapshot(const std::vector<result> & results) {
  std::vector<result> sorted = results;
  std::sort(sorted.begin(), sorted.end(), [](const result & a, const result & b) {
    return a.name < b.name;
  });

  for (const auto & entry : sorted) {
    std::printf("%s ns_per_op=%.3f iter=%" PRIu64 " runs=%zu\n",
                entry.name.c_str(),
                entry.ns_per_op,
                entry.iterations,
                entry.runs);
  }
}

void print_compare(const std::vector<result> & emel_results,
                   const std::vector<result> & reference_results) {
  std::vector<result> emel_sorted = emel_results;
  std::vector<result> ref_sorted = reference_results;

  std::sort(emel_sorted.begin(), emel_sorted.end(), [](const result & a, const result & b) {
    return a.name < b.name;
  });
  std::sort(ref_sorted.begin(), ref_sorted.end(), [](const result & a, const result & b) {
    return a.name < b.name;
  });

  const std::size_t count = std::min(emel_sorted.size(), ref_sorted.size());
  for (std::size_t i = 0; i < count; ++i) {
    const auto & emel_entry = emel_sorted[i];
    const auto & ref_entry = ref_sorted[i];
    if (emel_entry.name != ref_entry.name) {
      std::fprintf(stderr, "error: case mismatch %s vs %s\n",
                   emel_entry.name.c_str(), ref_entry.name.c_str());
      std::exit(1);
    }
    const double ratio = emel_entry.ns_per_op / ref_entry.ns_per_op;
    std::printf("%s emel.cpp %.3f ns/op, llama.cpp %.3f ns/op, ratio=%.3fx\n",
                emel_entry.name.c_str(),
                emel_entry.ns_per_op,
                ref_entry.ns_per_op,
                ratio);
  }

  if (emel_sorted.size() != ref_sorted.size()) {
    std::fprintf(stderr, "error: case count mismatch\n");
    std::exit(1);
  }
}

enum class mode {
  k_emel,
  k_reference,
  k_compare,
};

mode parse_mode(int argc, char ** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--mode=emel") {
      return mode::k_emel;
    }
    if (arg == "--mode=reference") {
      return mode::k_reference;
    }
    if (arg == "--mode=compare") {
      return mode::k_compare;
    }
  }
  return mode::k_emel;
}

}  // namespace

int main(int argc, char ** argv) {
  config cfg;
  cfg.iterations = read_env_u64("EMEL_BENCH_ITERS", k_default_iterations);
  cfg.runs = read_env_size("EMEL_BENCH_RUNS", k_default_runs);
  cfg.warmup_iterations = read_env_u64(
    "EMEL_BENCH_WARMUP_ITERS",
    std::min(cfg.iterations, k_default_warmup_iterations));
  cfg.warmup_runs = read_env_size("EMEL_BENCH_WARMUP_RUNS", k_default_warmup_runs);

  const mode run_mode = parse_mode(argc, argv);

  if (run_mode == mode::k_emel) {
    const auto results = run_emel_benchmarks(cfg);
    print_snapshot(results);
    return 0;
  }

  if (run_mode == mode::k_reference) {
    const auto results = run_reference_benchmarks(cfg);
    print_snapshot(results);
    return 0;
  }

  const auto emel_results = run_emel_benchmarks(cfg);
  const auto ref_results = run_reference_benchmarks(cfg);
  print_compare(emel_results, ref_results);
  return 0;
}
