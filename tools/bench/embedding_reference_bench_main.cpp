#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench_common.hpp"
#include "embedding_compare_contract.hpp"

namespace emel::bench {

void append_embedding_reference_cases(std::vector<result> & results,
                                      const config & cfg,
                                      std::vector<embedding_compare_record> * compare_records);
void print_embedding_reference_bench_metadata();

}  // namespace emel::bench

namespace {
namespace bench = emel::bench;

constexpr std::uint64_t k_default_iterations = 1;
constexpr std::size_t k_default_runs = 3;
constexpr std::uint64_t k_default_warmup_iterations = 1;
constexpr std::size_t k_default_warmup_runs = 1;
constexpr std::size_t k_max_runs = 10;

constexpr std::string_view k_embedding_reference_source =
#ifdef EMBEDDING_REFERENCE_SOURCE
  EMBEDDING_REFERENCE_SOURCE;
#else
  "unknown";
#endif

constexpr std::string_view k_embedding_reference_ref =
#ifdef EMBEDDING_REFERENCE_REF
  EMBEDDING_REFERENCE_REF;
#else
  "unknown";
#endif

constexpr std::string_view k_embedding_reference_repository =
#ifdef EMBEDDING_REFERENCE_REPOSITORY
  EMBEDDING_REFERENCE_REPOSITORY;
#else
  "unknown";
#endif

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
  if (parsed == 0u) {
    return fallback;
  }
  if (parsed > static_cast<std::uint64_t>(k_max_runs)) {
    return k_max_runs;
  }
  return static_cast<std::size_t>(parsed);
}

void print_results(const std::vector<bench::result> & results) {
  std::vector<bench::result> sorted = results;
  std::sort(sorted.begin(), sorted.end(), [](const bench::result & a, const bench::result & b) {
    return a.name < b.name;
  });
  for (const auto & entry : sorted) {
    std::printf("%s ns_per_op=%.3f prepare_ns=%.3f encode_ns=%.3f publish_ns=%.3f "
                "output_tokens=%" PRIu64 " output_dim=%" PRIu64
                " output_checksum=0x%016" PRIx64 " iter=%" PRIu64 " runs=%zu\n",
                entry.name.c_str(),
                entry.ns_per_op,
                entry.prepare_ns_per_op,
                entry.encode_ns_per_op,
                entry.publish_ns_per_op,
                entry.output_tokens,
                entry.output_dim,
                entry.output_checksum,
                entry.iterations,
                entry.runs);
  }
}

}  // namespace

int main() {
  bench::config cfg = {};
  cfg.iterations = read_env_u64("EMEL_BENCH_ITERS", k_default_iterations);
  cfg.runs = read_env_size("EMEL_BENCH_RUNS", k_default_runs);
  cfg.warmup_iterations = read_env_u64("EMEL_BENCH_WARMUP_ITERS", k_default_warmup_iterations);
  cfg.warmup_runs = read_env_size("EMEL_BENCH_WARMUP_RUNS", k_default_warmup_runs);

  const bool emit_jsonl = bench::embedding_compare_emit_jsonl();

  if (!emit_jsonl) {
    std::printf("# benchmark_config: iterations=%" PRIu64 " runs=%zu warmup_iterations=%" PRIu64
                " warmup_runs=%zu\n",
                cfg.iterations,
                cfg.runs,
                cfg.warmup_iterations,
                cfg.warmup_runs);
    std::printf("# embedding_reference_repository: %s\n", k_embedding_reference_repository.data());
    std::printf("# embedding_reference_ref: %s\n", k_embedding_reference_ref.data());
    std::printf("# embedding_reference_source: %s\n", k_embedding_reference_source.data());
    bench::print_embedding_reference_bench_metadata();
  }

  std::vector<bench::result> results = {};
  results.reserve(4u);
  std::vector<bench::embedding_compare_record> compare_records = {};
  bench::append_embedding_reference_cases(
    results,
    cfg,
    emit_jsonl ? &compare_records : nullptr);
  if (emit_jsonl) {
    for (auto & record : compare_records) {
      bench::maybe_dump_embedding_output(record);
      bench::print_embedding_compare_record_jsonl(record);
    }
    return 0;
  }
  print_results(results);
  return 0;
}
