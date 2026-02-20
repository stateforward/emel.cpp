#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace {

constexpr std::uint64_t k_default_iterations = 100000;
constexpr std::size_t k_default_runs = 5;
constexpr std::uint64_t k_default_warmup_iterations = 1000;
constexpr std::size_t k_default_warmup_runs = 1;
constexpr std::size_t k_max_runs = 25;

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

struct result {
  double ns_per_op = 0.0;
  std::uint64_t iterations = 0;
  std::size_t runs = 0;
};

using bench_fn = void (*)(ggml_gallocr_t, ggml_cgraph *, const int *, const int *);

result measure_case(
    ggml_gallocr_t galloc,
    ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids,
    std::uint64_t iterations,
    std::size_t runs,
    std::uint64_t warmup_iterations,
    std::size_t warmup_runs,
    bench_fn fn) {
  std::vector<double> samples;
  samples.reserve(runs);

  for (std::size_t run = 0; run < warmup_runs; ++run) {
    for (std::uint64_t i = 0; i < warmup_iterations; ++i) {
      fn(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    }
  }

  for (std::size_t run = 0; run < runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < iterations; ++i) {
      fn(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    }
    const auto end = std::chrono::steady_clock::now();
    const auto duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    samples.push_back(static_cast<double>(duration_ns) / static_cast<double>(iterations));
  }

  std::sort(samples.begin(), samples.end());
  const double median = samples[samples.size() / 2];

  result out;
  out.ns_per_op = median;
  out.iterations = iterations;
  out.runs = runs;
  return out;
}

void run_reserve(
    ggml_gallocr_t galloc,
    ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids) {
  (void)ggml_gallocr_reserve_n(galloc, graph, node_buffer_ids, leaf_buffer_ids);
}

void run_alloc(
    ggml_gallocr_t galloc,
    ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids) {
  (void)ggml_gallocr_alloc_graph(galloc, graph);
}

void run_full(
    ggml_gallocr_t galloc,
    ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids) {
  (void)ggml_gallocr_reserve_n(galloc, graph, node_buffer_ids, leaf_buffer_ids);
  (void)ggml_gallocr_alloc_graph(galloc, graph);
}

}  // namespace

int main() {
  const auto iterations = read_env_u64("EMEL_BENCH_ITERS", k_default_iterations);
  const auto runs = read_env_size("EMEL_BENCH_RUNS", k_default_runs);
  const auto warmup_iterations =
      read_env_u64("EMEL_BENCH_WARMUP_ITERS",
                   std::min(iterations, k_default_warmup_iterations));
  const auto warmup_runs = read_env_size("EMEL_BENCH_WARMUP_RUNS",
                                         k_default_warmup_runs);

  ggml_init_params params = {
    .mem_size = 4 * 1024 * 1024,
    .mem_buffer = nullptr,
    .no_alloc = true,
  };

  ggml_context * ctx = ggml_init(params);
  if (ctx == nullptr) {
    std::fprintf(stderr, "error: ggml_init failed\n");
    return 1;
  }

  ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
  if (input == nullptr) {
    std::fprintf(stderr, "error: tensor init failed\n");
    ggml_free(ctx);
    return 1;
  }
  ggml_set_input(input);

  ggml_tensor * node = ggml_dup(ctx, input);
  if (node == nullptr) {
    std::fprintf(stderr, "error: node init failed\n");
    ggml_free(ctx);
    return 1;
  }
  ggml_set_output(node);

  ggml_cgraph * graph = ggml_new_graph(ctx);
  if (graph == nullptr) {
    std::fprintf(stderr, "error: graph init failed\n");
    ggml_free(ctx);
    return 1;
  }
  ggml_build_forward_expand(graph, node);

  ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
  if (galloc == nullptr) {
    std::fprintf(stderr, "error: ggml_gallocr_new failed\n");
    ggml_free(ctx);
    return 1;
  }

  const int node_buffer_ids[1] = {0};
  const int leaf_buffer_ids[1] = {0};

  const result reserve_out =
      measure_case(galloc, graph, node_buffer_ids, leaf_buffer_ids, iterations, runs,
                   warmup_iterations, warmup_runs, run_reserve);
  const result alloc_out =
      measure_case(galloc, graph, node_buffer_ids, leaf_buffer_ids, iterations, runs,
                   warmup_iterations, warmup_runs, run_alloc);
  const result full_out =
      measure_case(galloc, graph, node_buffer_ids, leaf_buffer_ids, iterations, runs,
                   warmup_iterations, warmup_runs, run_full);

  std::printf("buffer/allocator_reserve_n ns_per_op=%.3f iter=%" PRIu64 " runs=%zu\n",
              reserve_out.ns_per_op,
              reserve_out.iterations,
              reserve_out.runs);
  std::printf("buffer/allocator_alloc_graph ns_per_op=%.3f iter=%" PRIu64 " runs=%zu\n",
              alloc_out.ns_per_op,
              alloc_out.iterations,
              alloc_out.runs);
  std::printf("buffer/allocator_full ns_per_op=%.3f iter=%" PRIu64 " runs=%zu\n",
              full_out.ns_per_op,
              full_out.iterations,
              full_out.runs);

  ggml_gallocr_free(galloc);
  ggml_free(ctx);

  return 0;
}
