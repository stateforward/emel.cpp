#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench_cases.hpp"
#include "bench_common.hpp"

namespace {
namespace bench = emel::bench;

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

std::vector<bench::result> run_emel_benchmarks(const bench::config & cfg) {
  std::vector<bench::result> results;
  results.reserve(10);
  bench::append_emel_buffer_allocator_cases(results, cfg);
  bench::append_emel_batch_splitter_cases(results, cfg);
  bench::append_emel_memory_coordinator_recurrent_cases(results, cfg);
  bench::append_emel_jinja_parser_cases(results, cfg);
  return results;
}

std::vector<bench::result> run_reference_benchmarks(const bench::config & cfg) {
  std::vector<bench::result> results;
  results.reserve(10);
  bench::append_reference_buffer_allocator_cases(results, cfg);
  bench::append_reference_batch_splitter_cases(results, cfg);
  bench::append_reference_memory_coordinator_recurrent_cases(results, cfg);
  bench::append_reference_jinja_parser_cases(results, cfg);
  return results;
}

void print_snapshot(const std::vector<bench::result> & results) {
  std::vector<bench::result> sorted = results;
  std::sort(sorted.begin(), sorted.end(), [](const bench::result & a, const bench::result & b) {
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

void print_compare(const std::vector<bench::result> & emel_results,
                   const std::vector<bench::result> & reference_results) {
  std::vector<bench::result> emel_sorted = emel_results;
  std::vector<bench::result> ref_sorted = reference_results;

  std::sort(emel_sorted.begin(), emel_sorted.end(), [](const bench::result & a,
                                                       const bench::result & b) {
    return a.name < b.name;
  });
  std::sort(ref_sorted.begin(), ref_sorted.end(), [](const bench::result & a,
                                                     const bench::result & b) {
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
  bench::config cfg;
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
