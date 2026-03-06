#include <algorithm>
#include <array>
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

constexpr bool k_host_is_x86_64 =
#if defined(__x86_64__) || defined(_M_X64)
  true;
#else
  false;
#endif

constexpr bool k_host_is_aarch64 =
#if defined(__aarch64__) || defined(_M_ARM64)
  true;
#else
  false;
#endif

bool case_supported_on_host(const bench::test_case & tc) {
  if (tc.append_emel == bench::append_emel_kernel_x86_64_cases ||
      tc.append_reference == bench::append_reference_kernel_x86_64_cases) {
    return k_host_is_x86_64;
  }
  if (tc.append_emel == bench::append_emel_kernel_aarch64_cases ||
      tc.append_reference == bench::append_reference_kernel_aarch64_cases) {
    return k_host_is_aarch64;
  }
  return true;
}

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

std::int32_t read_env_i32(const char * name, const std::int32_t fallback) {
  const char * value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char * end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value) {
    return fallback;
  }
  return static_cast<std::int32_t>(parsed);
}

constexpr bench::test_case make_test_case(const bench::append_case_fn emel_fn,
                                          const bench::append_case_fn reference_fn,
                                          const bool tokenizer_case = false) {
  return bench::test_case{
    .append_emel = emel_fn,
    .append_reference = reference_fn,
    .tokenizer_case = tokenizer_case,
  };
}

const auto & default_test_cases() {
  static const std::array<bench::test_case, 25> cases = {{
    make_test_case(bench::append_emel_batch_planner_cases,
                   bench::append_reference_batch_planner_cases),
    make_test_case(bench::append_emel_memory_kv_cases, bench::append_reference_memory_kv_cases),
    make_test_case(bench::append_emel_memory_recurrent_cases,
                   bench::append_reference_memory_recurrent_cases),
    make_test_case(bench::append_emel_memory_hybrid_cases,
                   bench::append_reference_memory_hybrid_cases),
    make_test_case(bench::append_emel_jinja_parser_cases,
                   bench::append_reference_jinja_parser_cases),
    make_test_case(bench::append_emel_jinja_formatter_cases,
                   bench::append_reference_jinja_formatter_cases),
    make_test_case(bench::append_emel_gbnf_rule_parser_cases,
                   bench::append_reference_gbnf_rule_parser_cases),
    make_test_case(bench::append_emel_logits_validator_cases,
                   bench::append_reference_logits_validator_cases),
    make_test_case(bench::append_emel_logits_sampler_cases,
                   bench::append_reference_logits_sampler_cases),
    make_test_case(bench::append_emel_kernel_x86_64_cases,
                   bench::append_reference_kernel_x86_64_cases),
    make_test_case(bench::append_emel_kernel_aarch64_cases,
                   bench::append_reference_kernel_aarch64_cases),
    make_test_case(bench::append_emel_sm_any_cases, bench::append_reference_sm_any_cases),
    make_test_case(bench::append_emel_tokenizer_preprocessor_bpe_cases,
                   bench::append_reference_tokenizer_preprocessor_bpe_cases),
    make_test_case(bench::append_emel_tokenizer_preprocessor_spm_cases,
                   bench::append_reference_tokenizer_preprocessor_spm_cases),
    make_test_case(bench::append_emel_tokenizer_preprocessor_ugm_cases,
                   bench::append_reference_tokenizer_preprocessor_ugm_cases),
    make_test_case(bench::append_emel_tokenizer_preprocessor_wpm_cases,
                   bench::append_reference_tokenizer_preprocessor_wpm_cases),
    make_test_case(bench::append_emel_tokenizer_preprocessor_rwkv_cases,
                   bench::append_reference_tokenizer_preprocessor_rwkv_cases),
    make_test_case(bench::append_emel_tokenizer_preprocessor_plamo2_cases,
                   bench::append_reference_tokenizer_preprocessor_plamo2_cases),
    make_test_case(bench::append_emel_encoder_bpe_cases, bench::append_reference_encoder_bpe_cases),
    make_test_case(bench::append_emel_encoder_spm_cases, bench::append_reference_encoder_spm_cases),
    make_test_case(bench::append_emel_encoder_wpm_cases, bench::append_reference_encoder_wpm_cases),
    make_test_case(bench::append_emel_encoder_ugm_cases, bench::append_reference_encoder_ugm_cases),
    make_test_case(bench::append_emel_encoder_rwkv_cases,
                   bench::append_reference_encoder_rwkv_cases),
    make_test_case(bench::append_emel_encoder_plamo2_cases,
                   bench::append_reference_encoder_plamo2_cases),
    make_test_case(bench::append_emel_encoder_fallback_cases,
                   bench::append_reference_encoder_fallback_cases),
  }};
  return cases;
}

const auto & kernel_test_cases() {
  static const std::array<bench::test_case, 2> cases = {{
    make_test_case(bench::append_emel_kernel_x86_64_cases,
                   bench::append_reference_kernel_x86_64_cases),
    make_test_case(bench::append_emel_kernel_aarch64_cases,
                   bench::append_reference_kernel_aarch64_cases),
  }};
  return cases;
}

template <size_t k_case_count>
std::vector<bench::result> run_benchmarks(const bench::config & cfg,
                                          const std::array<bench::test_case, k_case_count> & cases,
                                          const bool reference,
                                          const bool include_tokenizer) {
  std::vector<bench::result> results;
  results.reserve(k_case_count + 1);
  const std::int32_t selected_case_index = read_env_i32("EMEL_BENCH_CASE_INDEX", -1);

  std::size_t case_index = 0;
  for (const bench::test_case & tc : cases) {
    const bool selected_case = selected_case_index < 0 ||
        static_cast<std::int32_t>(case_index) == selected_case_index;
    case_index += 1;
    if (!selected_case) {
      continue;
    }
    if (tc.tokenizer_case && !include_tokenizer) {
      continue;
    }
    if (!case_supported_on_host(tc)) {
      continue;
    }
    bench::append_test_case(results, cfg, tc, reference);
  }

  if (include_tokenizer) {
    const bool selected_tokenizer = selected_case_index < 0 ||
      selected_case_index == static_cast<std::int32_t>(k_case_count);
    if (!selected_tokenizer) {
      return results;
    }
    const bench::test_case tokenizer_case = make_test_case(
      bench::append_emel_tokenizer_cases,
      bench::append_reference_tokenizer_cases,
      true);
    bench::append_test_case(results, cfg, tokenizer_case, reference);
  }
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
  k_kernel_emel,
  k_kernel_reference,
  k_kernel_compare,
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
    if (arg == "--mode=kernel-emel") {
      return mode::k_kernel_emel;
    }
    if (arg == "--mode=kernel-reference") {
      return mode::k_kernel_reference;
    }
    if (arg == "--mode=kernel-compare") {
      return mode::k_kernel_compare;
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

  if (run_mode == mode::k_kernel_emel) {
    const auto results = run_benchmarks(cfg, kernel_test_cases(), false, false);
    print_snapshot(results);
    return 0;
  }

  if (run_mode == mode::k_kernel_reference) {
    const auto results = run_benchmarks(cfg, kernel_test_cases(), true, false);
    print_snapshot(results);
    return 0;
  }

  if (run_mode == mode::k_kernel_compare) {
    const auto emel_results = run_benchmarks(cfg, kernel_test_cases(), false, false);
    const auto ref_results = run_benchmarks(cfg, kernel_test_cases(), true, false);
    print_compare(emel_results, ref_results);
    return 0;
  }

  if (run_mode == mode::k_emel) {
    const auto results = run_benchmarks(cfg, default_test_cases(), false, true);
    print_snapshot(results);
    return 0;
  }

  if (run_mode == mode::k_reference) {
    const auto results = run_benchmarks(cfg, default_test_cases(), true, true);
    print_snapshot(results);
    return 0;
  }

  const auto emel_results = run_benchmarks(cfg, default_test_cases(), false, true);
  const auto ref_results = run_benchmarks(cfg, default_test_cases(), true, true);
  print_compare(emel_results, ref_results);
  return 0;
}
