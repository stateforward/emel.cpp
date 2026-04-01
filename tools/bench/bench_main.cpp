#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <string>
#include <vector>

#include "bench_cases.hpp"
#include "bench_common.hpp"

namespace emel::bench {

bool generation_flash_evidence_ready() noexcept;
std::uint64_t generation_flash_evidence_dispatch_calls() noexcept;
std::uint64_t generation_flash_evidence_optimized_dispatch_calls() noexcept;
std::uint64_t generation_flash_evidence_shared_dispatch_calls() noexcept;
std::string_view generation_formatter_contract() noexcept;
std::uint32_t generation_runtime_contract_native_quantized_stage_count() noexcept;
std::uint32_t generation_runtime_contract_approved_dense_f32_stage_count() noexcept;
std::uint32_t generation_runtime_contract_disallowed_fallback_stage_count() noexcept;
std::uint32_t generation_runtime_contract_explicit_no_claim_stage_count() noexcept;
std::uint64_t generation_quantized_evidence_native_q8_0_dispatch_calls() noexcept;
std::uint64_t generation_quantized_evidence_packed_q8_0_dispatch_calls() noexcept;
std::uint64_t generation_quantized_evidence_optimized_q2_dispatch_calls() noexcept;
std::uint64_t generation_quantized_evidence_shared_q2_dispatch_calls() noexcept;
std::uint64_t generation_quantized_evidence_optimized_q3_dispatch_calls() noexcept;
std::uint64_t generation_quantized_evidence_shared_q3_dispatch_calls() noexcept;
std::uint64_t generation_quantized_evidence_optimized_q4_dispatch_calls() noexcept;
std::uint64_t generation_quantized_evidence_shared_q4_dispatch_calls() noexcept;
std::uint64_t generation_quantized_evidence_optimized_q6_dispatch_calls() noexcept;
std::uint64_t generation_quantized_evidence_shared_q6_dispatch_calls() noexcept;
std::int32_t generation_flash_evidence_emel_decode_calls() noexcept;
std::int32_t generation_flash_evidence_emel_logits_calls() noexcept;
std::int32_t generation_flash_evidence_reference_decode_calls() noexcept;
std::int32_t generation_flash_evidence_reference_logits_calls() noexcept;
std::string_view generation_architecture_contract() noexcept;

}  // namespace emel::bench

namespace {
namespace bench = emel::bench;

constexpr std::uint64_t k_default_iterations = 1000;
constexpr std::size_t k_default_runs = 3;
constexpr std::uint64_t k_default_warmup_iterations = 100;
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

constexpr std::string_view k_bench_reference_source =
#ifdef BENCH_REFERENCE_SOURCE
  BENCH_REFERENCE_SOURCE;
#else
  "unknown";
#endif

constexpr std::string_view k_bench_reference_ref =
#ifdef BENCH_REFERENCE_REF
  BENCH_REFERENCE_REF;
#else
  "unknown";
#endif

bool is_generation_case_name(const std::string & name) {
  return name.rfind("generation/preloaded_request/", 0u) == 0u;
}

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
                                          const std::string_view suite,
                                          const bool tokenizer_case = false) {
  return bench::test_case{
    .append_emel = emel_fn,
    .append_reference = reference_fn,
    .suite = suite,
    .tokenizer_case = tokenizer_case,
  };
}

const auto & default_test_cases() {
  static const std::array<bench::test_case, 27> cases = {{
    make_test_case(bench::append_emel_batch_planner_cases,
                   bench::append_reference_batch_planner_cases,
                   "batch_planner"),
    make_test_case(bench::append_emel_memory_kv_cases,
                   bench::append_reference_memory_kv_cases,
                   "memory_kv"),
    make_test_case(bench::append_emel_memory_recurrent_cases,
                   bench::append_reference_memory_recurrent_cases,
                   "memory_recurrent"),
    make_test_case(bench::append_emel_memory_hybrid_cases,
                   bench::append_reference_memory_hybrid_cases,
                   "memory_hybrid"),
    make_test_case(bench::append_emel_jinja_parser_cases,
                   bench::append_reference_jinja_parser_cases,
                   "jinja_parser"),
    make_test_case(bench::append_emel_jinja_formatter_cases,
                   bench::append_reference_jinja_formatter_cases,
                   "jinja_formatter"),
    make_test_case(bench::append_emel_gbnf_rule_parser_cases,
                   bench::append_reference_gbnf_rule_parser_cases,
                   "gbnf_rule_parser"),
    make_test_case(bench::append_emel_generation_cases,
                   bench::append_reference_generation_cases,
                   "generation"),
    make_test_case(bench::append_emel_flash_attention_cases,
                   bench::append_reference_flash_attention_cases,
                   "flash_attention"),
    make_test_case(bench::append_emel_logits_validator_cases,
                   bench::append_reference_logits_validator_cases,
                   "logits_validator"),
    make_test_case(bench::append_emel_logits_sampler_cases,
                   bench::append_reference_logits_sampler_cases,
                   "logits_sampler"),
    make_test_case(bench::append_emel_kernel_x86_64_cases,
                   bench::append_reference_kernel_x86_64_cases,
                   "kernel_x86_64"),
    make_test_case(bench::append_emel_kernel_aarch64_cases,
                   bench::append_reference_kernel_aarch64_cases,
                   "kernel_aarch64"),
    make_test_case(bench::append_emel_sm_any_cases,
                   bench::append_reference_sm_any_cases,
                   "sm_any"),
    make_test_case(bench::append_emel_tokenizer_preprocessor_bpe_cases,
                   bench::append_reference_tokenizer_preprocessor_bpe_cases,
                   "tokenizer_preprocessor_bpe"),
    make_test_case(bench::append_emel_tokenizer_preprocessor_spm_cases,
                   bench::append_reference_tokenizer_preprocessor_spm_cases,
                   "tokenizer_preprocessor_spm"),
    make_test_case(bench::append_emel_tokenizer_preprocessor_ugm_cases,
                   bench::append_reference_tokenizer_preprocessor_ugm_cases,
                   "tokenizer_preprocessor_ugm"),
    make_test_case(bench::append_emel_tokenizer_preprocessor_wpm_cases,
                   bench::append_reference_tokenizer_preprocessor_wpm_cases,
                   "tokenizer_preprocessor_wpm"),
    make_test_case(bench::append_emel_tokenizer_preprocessor_rwkv_cases,
                   bench::append_reference_tokenizer_preprocessor_rwkv_cases,
                   "tokenizer_preprocessor_rwkv"),
    make_test_case(bench::append_emel_tokenizer_preprocessor_plamo2_cases,
                   bench::append_reference_tokenizer_preprocessor_plamo2_cases,
                   "tokenizer_preprocessor_plamo2"),
    make_test_case(bench::append_emel_encoder_bpe_cases,
                   bench::append_reference_encoder_bpe_cases,
                   "encoder_bpe"),
    make_test_case(bench::append_emel_encoder_spm_cases,
                   bench::append_reference_encoder_spm_cases,
                   "encoder_spm"),
    make_test_case(bench::append_emel_encoder_wpm_cases,
                   bench::append_reference_encoder_wpm_cases,
                   "encoder_wpm"),
    make_test_case(bench::append_emel_encoder_ugm_cases,
                   bench::append_reference_encoder_ugm_cases,
                   "encoder_ugm"),
    make_test_case(bench::append_emel_encoder_rwkv_cases,
                   bench::append_reference_encoder_rwkv_cases,
                   "encoder_rwkv"),
    make_test_case(bench::append_emel_encoder_plamo2_cases,
                   bench::append_reference_encoder_plamo2_cases,
                   "encoder_plamo2"),
    make_test_case(bench::append_emel_encoder_fallback_cases,
                   bench::append_reference_encoder_fallback_cases,
                   "encoder_fallback"),
  }};
  return cases;
}

const auto & kernel_test_cases() {
  static const std::array<bench::test_case, 2> cases = {{
    make_test_case(bench::append_emel_kernel_x86_64_cases,
                   bench::append_reference_kernel_x86_64_cases,
                   "kernel_x86_64"),
    make_test_case(bench::append_emel_kernel_aarch64_cases,
                   bench::append_reference_kernel_aarch64_cases,
                   "kernel_aarch64"),
  }};
  return cases;
}

void print_benchmark_config(const bench::config & cfg) {
  const auto generation_iterations = read_env_u64("EMEL_BENCH_GENERATION_ITERS", 1u);
  const auto generation_runs = read_env_size("EMEL_BENCH_GENERATION_RUNS", cfg.runs);
  const auto generation_warmup_iterations =
      read_env_u64("EMEL_BENCH_GENERATION_WARMUP_ITERS", 0u);
  const auto generation_warmup_runs =
      read_env_size("EMEL_BENCH_GENERATION_WARMUP_RUNS", 0u);
  std::printf("# benchmark_config: iterations=%" PRIu64
              " runs=%zu warmup_iterations=%" PRIu64
              " warmup_runs=%zu generation_iterations=%" PRIu64
              " generation_runs=%zu generation_warmup_iterations=%" PRIu64
              " generation_warmup_runs=%zu\n",
              cfg.iterations,
              cfg.runs,
              cfg.warmup_iterations,
              cfg.warmup_runs,
              generation_iterations,
              generation_runs,
              generation_warmup_iterations,
              generation_warmup_runs);
}

template <size_t k_case_count>
std::vector<bench::result> run_benchmarks(const bench::config & cfg,
                                          const std::array<bench::test_case, k_case_count> & cases,
                                          const bool reference,
                                          const bool include_tokenizer) {
  std::vector<bench::result> results;
  results.reserve(k_case_count + 1);
  const std::int32_t selected_case_index = read_env_i32("EMEL_BENCH_CASE_INDEX", -1);
  const char * selected_suite = std::getenv("EMEL_BENCH_SUITE");
  const bool filter_by_suite = selected_suite != nullptr && selected_suite[0] != '\0';
  bool suite_seen = false;

  std::size_t case_index = 0;
  for (const bench::test_case & tc : cases) {
    const bool selected_case = selected_case_index < 0 ||
        static_cast<std::int32_t>(case_index) == selected_case_index;
    const bool selected_suite_case = !filter_by_suite || tc.suite == selected_suite;
    suite_seen = suite_seen || selected_suite_case;
    case_index += 1;
    if (!selected_case || !selected_suite_case) {
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
    const bool selected_tokenizer_suite = !filter_by_suite ||
      std::string_view{"tokenizer"} == selected_suite;
    suite_seen = suite_seen || selected_tokenizer_suite;
    if (!selected_tokenizer) {
      if (filter_by_suite && !suite_seen) {
        std::fprintf(stderr, "error: unknown benchmark suite '%s'\n", selected_suite);
        std::exit(1);
      }
      return results;
    }
    const bench::test_case tokenizer_case = make_test_case(
      bench::append_emel_tokenizer_cases,
      bench::append_reference_tokenizer_cases,
      "tokenizer",
      true);
    if (selected_tokenizer_suite) {
      bench::append_test_case(results, cfg, tokenizer_case, reference);
    }
  }

  if (filter_by_suite && !suite_seen) {
    std::fprintf(stderr, "error: unknown benchmark suite '%s'\n", selected_suite);
    std::exit(1);
  }
  return results;
}

void print_snapshot(const std::vector<bench::result> & results, const bench::config & cfg) {
  std::vector<bench::result> sorted = results;
  std::sort(sorted.begin(), sorted.end(), [](const bench::result & a, const bench::result & b) {
    return a.name < b.name;
  });

  print_benchmark_config(cfg);
  for (const auto & entry : sorted) {
    std::printf("%s ns_per_op=%.3f iter=%" PRIu64 " runs=%zu\n",
                entry.name.c_str(),
                entry.ns_per_op,
                entry.iterations,
                entry.runs);
  }
}

void print_compare(const std::vector<bench::result> & emel_results,
                   const std::vector<bench::result> & reference_results,
                   const bench::config & cfg) {
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

  const auto duplicate_emel = std::adjacent_find(
    emel_sorted.begin(), emel_sorted.end(), [](const bench::result & a, const bench::result & b) {
      return a.name == b.name;
    });
  if (duplicate_emel != emel_sorted.end()) {
    std::fprintf(stderr, "error: duplicate emel case %s\n", duplicate_emel->name.c_str());
    std::exit(1);
  }

  const auto duplicate_ref = std::adjacent_find(
    ref_sorted.begin(), ref_sorted.end(), [](const bench::result & a, const bench::result & b) {
      return a.name == b.name;
    });
  if (duplicate_ref != ref_sorted.end()) {
    std::fprintf(stderr, "error: duplicate reference case %s\n", duplicate_ref->name.c_str());
    std::exit(1);
  }

  const auto generation_emel = std::find_if(
    emel_sorted.begin(), emel_sorted.end(), [](const bench::result & entry) {
      return entry.name == bench::k_generation_case_name;
    });
  const auto generation_ref = std::find_if(
    ref_sorted.begin(), ref_sorted.end(), [](const bench::result & entry) {
      return entry.name == bench::k_generation_case_name;
    });
  const bool any_generation_emel =
      std::any_of(emel_sorted.begin(), emel_sorted.end(), [](const bench::result & entry) {
        return is_generation_case_name(entry.name);
      });
  const bool any_generation_ref =
      std::any_of(ref_sorted.begin(), ref_sorted.end(), [](const bench::result & entry) {
        return is_generation_case_name(entry.name);
      });
  if (any_generation_emel != any_generation_ref) {
    std::fprintf(stderr, "error: generation suite mismatch between emel and reference\n");
    std::exit(1);
  }
  const bool generation_present = generation_emel != emel_sorted.end() ||
      generation_ref != ref_sorted.end();
  if ((any_generation_emel || any_generation_ref) && !generation_present) {
    std::fprintf(stderr,
                 "error: missing current maintained generation case %.*s\n",
                 static_cast<int>(bench::k_generation_case_name.size()),
                 bench::k_generation_case_name.data());
    std::exit(1);
  }
  if ((generation_emel == emel_sorted.end()) != (generation_ref == ref_sorted.end())) {
    std::fprintf(stderr, "error: generation case mismatch between emel and reference\n");
    std::exit(1);
  }
  if (generation_present && generation_emel == emel_sorted.end()) {
    std::fprintf(stderr, "error: missing emel generation case %.*s\n",
                 static_cast<int>(bench::k_generation_case_name.size()),
                 bench::k_generation_case_name.data());
    std::exit(1);
  }
  if (generation_present && generation_ref == ref_sorted.end()) {
    std::fprintf(stderr, "error: missing reference generation case %.*s\n",
                 static_cast<int>(bench::k_generation_case_name.size()),
                 bench::k_generation_case_name.data());
    std::exit(1);
  }

  if (emel_sorted.size() != ref_sorted.size()) {
    std::fprintf(stderr, "error: case count mismatch emel=%zu reference=%zu\n",
                 emel_sorted.size(),
                 ref_sorted.size());
    std::exit(1);
  }

  std::printf("# reference_impl: source=%.*s ref=%.*s\n",
              static_cast<int>(k_bench_reference_source.size()),
              k_bench_reference_source.data(),
              static_cast<int>(k_bench_reference_ref.size()),
              k_bench_reference_ref.data());
  print_benchmark_config(cfg);
  if (generation_present) {
    const std::string_view formatter_contract = bench::generation_formatter_contract();
    const std::string_view architecture_contract = bench::generation_architecture_contract();
    if (!architecture_contract.empty()) {
      std::printf("# generation_architecture: %.*s\n",
                  static_cast<int>(architecture_contract.size()),
                  architecture_contract.data());
    }
    if (!formatter_contract.empty()) {
      std::printf("# generation_formatter_contract: %.*s\n",
                  static_cast<int>(formatter_contract.size()),
                  formatter_contract.data());
    }
  }

  if (generation_present) {
    if (!bench::generation_flash_evidence_ready()) {
      std::fprintf(stderr, "error: missing generation runtime evidence\n");
      std::exit(1);
    }

    const auto flash_dispatch_calls = bench::generation_flash_evidence_dispatch_calls();
    const auto optimized_flash_dispatch_calls =
        bench::generation_flash_evidence_optimized_dispatch_calls();
    const auto shared_flash_dispatch_calls =
        bench::generation_flash_evidence_shared_dispatch_calls();
    const auto native_quantized_stage_count =
        bench::generation_runtime_contract_native_quantized_stage_count();
    const auto approved_dense_f32_stage_count =
        bench::generation_runtime_contract_approved_dense_f32_stage_count();
    const auto disallowed_fallback_stage_count =
        bench::generation_runtime_contract_disallowed_fallback_stage_count();
    const auto explicit_no_claim_stage_count =
        bench::generation_runtime_contract_explicit_no_claim_stage_count();
    const auto native_q8_0_dispatch_calls =
        bench::generation_quantized_evidence_native_q8_0_dispatch_calls();
    const auto packed_q8_0_dispatch_calls =
        bench::generation_quantized_evidence_packed_q8_0_dispatch_calls();
    const auto optimized_q2_dispatch_calls =
        bench::generation_quantized_evidence_optimized_q2_dispatch_calls();
    const auto shared_q2_dispatch_calls =
        bench::generation_quantized_evidence_shared_q2_dispatch_calls();
    const auto optimized_q3_dispatch_calls =
        bench::generation_quantized_evidence_optimized_q3_dispatch_calls();
    const auto shared_q3_dispatch_calls =
        bench::generation_quantized_evidence_shared_q3_dispatch_calls();
    const auto optimized_q4_dispatch_calls =
        bench::generation_quantized_evidence_optimized_q4_dispatch_calls();
    const auto shared_q4_dispatch_calls =
        bench::generation_quantized_evidence_shared_q4_dispatch_calls();
    const auto optimized_q6_dispatch_calls =
        bench::generation_quantized_evidence_optimized_q6_dispatch_calls();
    const auto shared_q6_dispatch_calls =
        bench::generation_quantized_evidence_shared_q6_dispatch_calls();
    const bool is_lfm2_generation =
        bench::generation_architecture_contract() == std::string_view{"lfm2"};
    const auto emel_decode_calls = bench::generation_flash_evidence_emel_decode_calls();
    const auto emel_logits_calls = bench::generation_flash_evidence_emel_logits_calls();
    const auto reference_decode_calls = bench::generation_flash_evidence_reference_decode_calls();
    const auto reference_logits_calls = bench::generation_flash_evidence_reference_logits_calls();
    if (emel_decode_calls != 0 || emel_logits_calls != 0 ||
        reference_decode_calls != 0 || reference_logits_calls != 0) {
      std::fprintf(stderr,
                   "error: invalid generation runtime evidence flash_dispatch_calls=%" PRIu64
                   " optimized_flash_dispatch_calls=%" PRIu64
                   " shared_flash_dispatch_calls=%" PRIu64
                   " emel_decode_calls=%d emel_logits_calls=%d reference_decode_calls=%d "
                   "reference_logits_calls=%d\n",
                   flash_dispatch_calls,
                   optimized_flash_dispatch_calls,
                   shared_flash_dispatch_calls,
                   emel_decode_calls,
                   emel_logits_calls,
                   reference_decode_calls,
                   reference_logits_calls);
      std::exit(1);
    }
    if (flash_dispatch_calls == 0 &&
        (optimized_flash_dispatch_calls != 0 || shared_flash_dispatch_calls != 0)) {
      std::fprintf(stderr,
                   "error: invalid zero-flash attribution optimized_flash_dispatch_calls=%" PRIu64
                   " shared_flash_dispatch_calls=%" PRIu64 "\n",
                   optimized_flash_dispatch_calls,
                   shared_flash_dispatch_calls);
      std::exit(1);
    }
    if (k_host_is_aarch64 && flash_dispatch_calls == 0) {
      std::fprintf(stderr,
                   "error: missing ARM flash attribution flash_dispatch_calls=%" PRIu64 "\n",
                   flash_dispatch_calls);
      std::exit(1);
    }
    if (flash_dispatch_calls != 0 && k_host_is_aarch64 &&
        (optimized_flash_dispatch_calls == 0 || shared_flash_dispatch_calls != 0)) {
      std::fprintf(stderr,
                   "error: invalid ARM flash attribution optimized_flash_dispatch_calls=%" PRIu64
                   " shared_flash_dispatch_calls=%" PRIu64 "\n",
                   optimized_flash_dispatch_calls,
                   shared_flash_dispatch_calls);
      std::exit(1);
    }
    if (flash_dispatch_calls != 0 && !k_host_is_aarch64 &&
        (optimized_flash_dispatch_calls != 0 || shared_flash_dispatch_calls != 0)) {
      std::fprintf(stderr,
                   "error: invalid non-ARM flash attribution optimized_flash_dispatch_calls=%" PRIu64
                   " shared_flash_dispatch_calls=%" PRIu64 "\n",
                   optimized_flash_dispatch_calls,
                   shared_flash_dispatch_calls);
      std::exit(1);
    }
    const bool invalid_lfm2_quantized_evidence =
        native_q8_0_dispatch_calls != 0 || packed_q8_0_dispatch_calls != 0 ||
        optimized_q2_dispatch_calls != 0 || shared_q2_dispatch_calls != 0 ||
        optimized_q3_dispatch_calls != 0 || shared_q3_dispatch_calls != 0 ||
        optimized_q4_dispatch_calls == 0 || shared_q4_dispatch_calls != 0 ||
        optimized_q6_dispatch_calls == 0 || shared_q6_dispatch_calls != 0;
    const bool invalid_default_quantized_evidence =
        (native_q8_0_dispatch_calls + packed_q8_0_dispatch_calls) == 0 ||
        optimized_q2_dispatch_calls != 0 || shared_q2_dispatch_calls != 0 ||
        optimized_q3_dispatch_calls != 0 || shared_q3_dispatch_calls != 0 ||
        optimized_q4_dispatch_calls != 0 || shared_q4_dispatch_calls != 0 ||
        optimized_q6_dispatch_calls != 0 || shared_q6_dispatch_calls != 0;
    if ((is_lfm2_generation && invalid_lfm2_quantized_evidence) ||
        (!is_lfm2_generation && invalid_default_quantized_evidence)) {
      std::fprintf(stderr,
                   "error: invalid generation quantized evidence native_q8_0_dispatch_calls=%" PRIu64
                   " packed_q8_0_dispatch_calls=%" PRIu64
                   " optimized_q2_dispatch_calls=%" PRIu64
                   " shared_q2_dispatch_calls=%" PRIu64
                   " optimized_q3_dispatch_calls=%" PRIu64
                   " shared_q3_dispatch_calls=%" PRIu64
                   " optimized_q4_dispatch_calls=%" PRIu64
                   " shared_q4_dispatch_calls=%" PRIu64
                   " optimized_q6_dispatch_calls=%" PRIu64
                   " shared_q6_dispatch_calls=%" PRIu64 "\n",
                   native_q8_0_dispatch_calls,
                   packed_q8_0_dispatch_calls,
                   optimized_q2_dispatch_calls,
                   shared_q2_dispatch_calls,
                   optimized_q3_dispatch_calls,
                   shared_q3_dispatch_calls,
                   optimized_q4_dispatch_calls,
                   shared_q4_dispatch_calls,
                   optimized_q6_dispatch_calls,
                   shared_q6_dispatch_calls);
      std::exit(1);
    }
    if (native_quantized_stage_count != 8u || approved_dense_f32_stage_count != 6u ||
        disallowed_fallback_stage_count != 0u || explicit_no_claim_stage_count != 0u) {
      std::fprintf(stderr,
                   "error: invalid generation runtime contract native_quantized=%" PRIu32
                   " approved_dense_f32_by_contract=%" PRIu32
                   " disallowed_fallback=%" PRIu32
                   " explicit_no_claim=%" PRIu32 "\n",
                native_quantized_stage_count,
                approved_dense_f32_stage_count,
                disallowed_fallback_stage_count,
                explicit_no_claim_stage_count);
      std::exit(1);
    }

    std::printf("# generation_flash_evidence: case=%.*s flash_dispatch_calls=%" PRIu64
                " optimized_flash_dispatch_calls=%" PRIu64
                " shared_flash_dispatch_calls=%" PRIu64
                " emel_decode_calls=%d emel_logits_calls=%d reference_decode_calls=%d "
                "reference_logits_calls=%d\n",
                static_cast<int>(bench::k_generation_case_name.size()),
                bench::k_generation_case_name.data(),
                flash_dispatch_calls,
                optimized_flash_dispatch_calls,
                shared_flash_dispatch_calls,
                emel_decode_calls,
                emel_logits_calls,
                reference_decode_calls,
                reference_logits_calls);
    std::printf("# generation_runtime_contract: case=%.*s native_quantized=%" PRIu32
                " approved_dense_f32_by_contract=%" PRIu32
                " disallowed_fallback=%" PRIu32
                " explicit_no_claim=%" PRIu32 "\n",
                static_cast<int>(bench::k_generation_case_name.size()),
                bench::k_generation_case_name.data(),
                native_quantized_stage_count,
                approved_dense_f32_stage_count,
                disallowed_fallback_stage_count,
                explicit_no_claim_stage_count);
    std::printf("# generation_quantized_evidence: case=%.*s native_q8_0_dispatch_calls=%" PRIu64
                " packed_q8_0_dispatch_calls=%" PRIu64
                " optimized_q2_dispatch_calls=%" PRIu64
                " shared_q2_dispatch_calls=%" PRIu64
                " optimized_q3_dispatch_calls=%" PRIu64
                " shared_q3_dispatch_calls=%" PRIu64
                " optimized_q4_dispatch_calls=%" PRIu64
                " shared_q4_dispatch_calls=%" PRIu64
                " optimized_q6_dispatch_calls=%" PRIu64
                " shared_q6_dispatch_calls=%" PRIu64 "\n",
                static_cast<int>(bench::k_generation_case_name.size()),
                bench::k_generation_case_name.data(),
                native_q8_0_dispatch_calls,
                packed_q8_0_dispatch_calls,
                optimized_q2_dispatch_calls,
                shared_q2_dispatch_calls,
                optimized_q3_dispatch_calls,
                shared_q3_dispatch_calls,
                optimized_q4_dispatch_calls,
                shared_q4_dispatch_calls,
                optimized_q6_dispatch_calls,
                shared_q6_dispatch_calls);
  }

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
    print_snapshot(results, cfg);
    return 0;
  }

  if (run_mode == mode::k_kernel_reference) {
    const auto results = run_benchmarks(cfg, kernel_test_cases(), true, false);
    print_snapshot(results, cfg);
    return 0;
  }

  if (run_mode == mode::k_kernel_compare) {
    const auto emel_results = run_benchmarks(cfg, kernel_test_cases(), false, false);
    const auto ref_results = run_benchmarks(cfg, kernel_test_cases(), true, false);
    print_compare(emel_results, ref_results, cfg);
    return 0;
  }

  if (run_mode == mode::k_emel) {
    const auto results = run_benchmarks(cfg, default_test_cases(), false, true);
    print_snapshot(results, cfg);
    return 0;
  }

  if (run_mode == mode::k_reference) {
    const auto results = run_benchmarks(cfg, default_test_cases(), true, true);
    print_snapshot(results, cfg);
    return 0;
  }

  const auto emel_results = run_benchmarks(cfg, default_test_cases(), false, true);
  const auto ref_results = run_benchmarks(cfg, default_test_cases(), true, true);
  print_compare(emel_results, ref_results, cfg);
  return 0;
}
