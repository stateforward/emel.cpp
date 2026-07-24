#include "../generation_fixture_registry.hpp"
#include "../generation_formatter_contract.hpp"
#include "../generation_route_policy.hpp"
#include "bench_cases.hpp"
#include "embedding_generator_bench_helpers.hpp"
#include "generation_compare_contract.hpp"
#include "generation_workload_manifest.hpp"
#include "model_load_strategy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/gguf/loader/any.hpp"
#include "emel/gguf/loader/errors.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/io/events.hpp"
#include "emel/io/read/sm.hpp"
#include "emel/io/source/any.hpp"
#include "emel/io/staged_read/sm.hpp"
#include "emel/logits/sampler/events.hpp"
#include "emel/memory/view.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/generation/any.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/tensor/errors.hpp"
#include "emel/model/tensor/events.hpp"
#include "emel/model/tensor/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/generator/errors.hpp"
#include "emel/text/generator/events.hpp"
#include "emel/text/generator/sm.hpp"
#include "emel/text/renderer/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

#include "ggml.h"
#include "llama-context.h"
#include "llama-memory.h"
#include "llama-vocab.h"
#include "llama.h"

namespace {

constexpr size_t k_generation_output_capacity = 65536u;
constexpr size_t k_generation_token_trace_capacity = 4096u;

struct generation_case_spec {
  std::string name = {};
  std::string prompt = {};
  int32_t max_tokens = 0;
  emel::bench::generation_workload_manifest manifest = {};
};

struct generation_fixture_spec {
  const emel::tools::generation_fixture_registry::maintained_fixture *fixture =
      nullptr;
};

constexpr emel::tools::generation_fixture_registry::maintained_fixture
    k_gemma4_emel_generation_fixture = {
        .name = "gemma-4-e2b-it-Q8_0.gguf",
        .slug = "gemma_4_e2b_it_q8_0",
        .fixture_rel = "tests/models/gemma-4-e2b-it-Q8_0.gguf",
        .current_publication = false,
};

// 4B-class scale fixture: q4_K_M Qwen3, same architecture family as the 0.6B
// qwen3 fixture. Used to exercise memory-bandwidth / lane-scaling / KV pressure
// at scale. Bench-only (like gemma4 above): defined locally rather than in the
// shared parity registry so it is NOT subject to the maintained append-only
// parity baseline / drift contract. Not a publication baseline; the binary is
// not committed (see tests/models/.gitignore and tests/models/README.md
// provenance).
constexpr emel::tools::generation_fixture_registry::maintained_fixture
    k_qwen3_4b_emel_generation_fixture = {
        .name = "Qwen3-4B-Q4_K_M.gguf",
        .slug = "qwen3_4b_q4_k_m",
        .fixture_rel = "tests/models/Qwen3-4B-Q4_K_M.gguf",
        .current_publication = false,
};

constexpr generation_fixture_spec k_qwen3_generation_fixture = {
    .fixture =
        &emel::tools::generation_fixture_registry::k_qwen3_generation_fixture,
};

constexpr generation_fixture_spec k_lfm2_generation_fixture = {
    .fixture =
        &emel::tools::generation_fixture_registry::k_lfm2_generation_fixture,
};

constexpr generation_fixture_spec k_gemma4_generation_fixture = {
    .fixture = &k_gemma4_emel_generation_fixture,
};

constexpr generation_fixture_spec k_lfm2_230m_generation_fixture = {
    .fixture = &emel::tools::generation_fixture_registry::
                   k_lfm2_230m_generation_fixture,
};

constexpr generation_fixture_spec k_qwen3_4b_generation_fixture = {
    .fixture = &k_qwen3_4b_emel_generation_fixture,
};

constexpr std::array<generation_fixture_spec, 4> k_compare_generation_fixtures =
    {
        k_qwen3_generation_fixture,
        k_lfm2_generation_fixture,
        k_lfm2_230m_generation_fixture,
        k_qwen3_4b_generation_fixture,
};

constexpr std::array<generation_fixture_spec, 5> k_emel_generation_fixtures = {
    k_qwen3_generation_fixture,    k_lfm2_generation_fixture,
    k_gemma4_generation_fixture,   k_lfm2_230m_generation_fixture,
    k_qwen3_4b_generation_fixture,
};

using llama_model_ptr =
    std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using llama_context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

constexpr llama_flash_attn_type k_reference_flash_attn_type =
    LLAMA_FLASH_ATTN_TYPE_ENABLED;
constexpr int32_t k_generation_multithreaded_thread_count = 8;
constexpr int32_t k_generation_single_thread_count = 1;
constexpr char k_generation_benchmark_lane_env[] = "EMEL_BENCH_GENERATION_LANE";
constexpr char k_generation_benchmark_lanes_env[] =
    "EMEL_BENCH_GENERATION_LANES";
constexpr char k_generation_reference_threads_env[] =
    "EMEL_BENCH_GENERATION_REFERENCE_THREADS";
constexpr char k_generation_reference_decode_threads_env[] =
    "EMEL_BENCH_GENERATION_REFERENCE_DECODE_THREADS";
constexpr char k_generation_reference_batch_threads_env[] =
    "EMEL_BENCH_GENERATION_REFERENCE_BATCH_THREADS";
constexpr char k_legacy_generation_reference_threads_env[] =
    "EMEL_BENCH_REFERENCE_THREADS";
constexpr char k_generation_stage_probe_env[] = "EMEL_GENERATION_STAGE_PROBE";
constexpr std::string_view k_generation_benchmark_lane_single = "single";
constexpr std::string_view k_generation_benchmark_lane_multithreaded =
    "multithreaded";
constexpr std::string_view k_generation_multithreaded_thread_contract =
    "emel_parallel_matmul_lanes=8";
constexpr std::string_view k_generation_single_thread_contract =
    "emel_serial_matmul_lanes=1";

enum class generation_benchmark_lane : uint8_t {
  single,
  multithreaded,
};

enum class generation_stage_probe_selection : uint8_t {
  off,
  publication,
  selected,
};

struct generation_benchmark_lane_selection {
  std::array<generation_benchmark_lane, 2> lanes = {
      generation_benchmark_lane::single,
      generation_benchmark_lane::multithreaded,
  };
  size_t count = 2u;
};

generation_benchmark_lane g_generation_benchmark_lane_override =
    generation_benchmark_lane::multithreaded;
bool g_generation_benchmark_lane_override_set = false;

generation_benchmark_lane
parse_generation_benchmark_lane(std::string_view value, const char *name) {
  if (value == k_generation_benchmark_lane_single) {
    return generation_benchmark_lane::single;
  }
  if (value == k_generation_benchmark_lane_multithreaded) {
    return generation_benchmark_lane::multithreaded;
  }
  std::fprintf(stderr, "error: %s must be 'single' or 'multithreaded'\n", name);
  std::exit(1);
}

generation_benchmark_lane current_generation_benchmark_lane() {
  if (g_generation_benchmark_lane_override_set) {
    return g_generation_benchmark_lane_override;
  }
  const char *value = std::getenv(k_generation_benchmark_lane_env);
  if (value == nullptr || value[0] == '\0') {
    return generation_benchmark_lane::multithreaded;
  }
  return parse_generation_benchmark_lane(value,
                                         k_generation_benchmark_lane_env);
}

generation_benchmark_lane_selection selected_generation_benchmark_lanes() {
  const char *lane_value = std::getenv(k_generation_benchmark_lane_env);
  if (lane_value != nullptr && lane_value[0] != '\0') {
    return generation_benchmark_lane_selection{
        {parse_generation_benchmark_lane(lane_value,
                                         k_generation_benchmark_lane_env),
         generation_benchmark_lane::multithreaded},
        1u};
  }

  const char *lanes_value = std::getenv(k_generation_benchmark_lanes_env);
  if (lanes_value == nullptr || lanes_value[0] == '\0') {
    return {};
  }
  const std::string_view parsed{lanes_value};
  if (parsed == "both" || parsed == "single,multithreaded" ||
      parsed == "multithreaded,single") {
    return {};
  }
  return generation_benchmark_lane_selection{
      {parse_generation_benchmark_lane(parsed,
                                       k_generation_benchmark_lanes_env),
       generation_benchmark_lane::multithreaded},
      1u};
}

struct scoped_generation_benchmark_lane {
  explicit scoped_generation_benchmark_lane(
      generation_benchmark_lane lane) noexcept
      : previous(g_generation_benchmark_lane_override),
        previous_set(g_generation_benchmark_lane_override_set) {
    g_generation_benchmark_lane_override = lane;
    g_generation_benchmark_lane_override_set = true;
  }

  ~scoped_generation_benchmark_lane() {
    g_generation_benchmark_lane_override = previous;
    g_generation_benchmark_lane_override_set = previous_set;
  }

  generation_benchmark_lane previous;
  bool previous_set;
};

std::string_view generation_benchmark_lane_name() {
  return current_generation_benchmark_lane() ==
                 generation_benchmark_lane::single
             ? k_generation_benchmark_lane_single
             : k_generation_benchmark_lane_multithreaded;
}

emel::text::generator::benchmark_lane generator_benchmark_lane_event_value() {
  return current_generation_benchmark_lane() ==
                 generation_benchmark_lane::single
             ? emel::text::generator::benchmark_lane::single
             : emel::text::generator::benchmark_lane::multithreaded;
}

int32_t generation_emel_thread_count() {
  return current_generation_benchmark_lane() ==
                 generation_benchmark_lane::single
             ? k_generation_single_thread_count
             : k_generation_multithreaded_thread_count;
}

std::string_view generation_emel_thread_contract() {
  return current_generation_benchmark_lane() ==
                 generation_benchmark_lane::single
             ? k_generation_single_thread_contract
             : k_generation_multithreaded_thread_contract;
}

std::string generation_benchmark_case_name(std::string_view case_name) {
  std::string out{case_name};
  out += "/";
  out += generation_benchmark_lane_name();
  return out;
}

int32_t read_env_i32_positive(const char *name, const int32_t fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }

  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0L ||
      parsed > static_cast<long>(std::numeric_limits<int32_t>::max())) {
    return fallback;
  }
  return static_cast<int32_t>(parsed);
}

int32_t generation_reference_default_thread_count() {
  if (current_generation_benchmark_lane() ==
      generation_benchmark_lane::single) {
    return k_generation_single_thread_count;
  }
  const int32_t fallback = k_generation_multithreaded_thread_count;
  if (std::getenv(k_generation_reference_threads_env) != nullptr) {
    return read_env_i32_positive(k_generation_reference_threads_env, fallback);
  }
  return read_env_i32_positive(k_legacy_generation_reference_threads_env,
                               fallback);
}

int32_t generation_reference_decode_thread_count() {
  return read_env_i32_positive(k_generation_reference_decode_threads_env,
                               generation_reference_default_thread_count());
}

int32_t generation_reference_batch_thread_count() {
  return read_env_i32_positive(k_generation_reference_batch_threads_env,
                               generation_reference_default_thread_count());
}

int32_t generation_reference_thread_count() {
  return generation_reference_decode_thread_count();
}

std::string generation_reference_thread_contract(const int32_t decode_threads,
                                                 const int32_t batch_threads) {
  return "llama.cpp_n_threads=" + std::to_string(decode_threads) +
         ",n_threads_batch=" + std::to_string(batch_threads);
}

std::string generation_reference_thread_contract() {
  return generation_reference_thread_contract(
      generation_reference_decode_thread_count(),
      generation_reference_batch_thread_count());
}

std::uint64_t read_env_u64(const char *name, const std::uint64_t fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }

  char *end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value) {
    return fallback;
  }
  return static_cast<std::uint64_t>(parsed);
}

std::size_t read_env_size(const char *name, const std::size_t fallback) {
  const auto parsed = read_env_u64(name, static_cast<std::uint64_t>(fallback));
  return parsed == 0u ? fallback : static_cast<std::size_t>(parsed);
}

bool env_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

std::string_view generation_workload_filter() {
  const char *value = std::getenv("EMEL_GENERATION_WORKLOAD_ID");
  if (value == nullptr || value[0] == '\0') {
    return {};
  }
  return value;
}

bool generation_workload_selected(const generation_case_spec &spec) {
  const std::string_view filter = generation_workload_filter();
  if (filter.empty() || filter == "all") {
    return true;
  }
  return spec.manifest.id == filter || spec.name == filter ||
         spec.manifest.compare_group == filter;
}

generation_stage_probe_selection generation_stage_probe_mode() {
  const char *value = std::getenv(k_generation_stage_probe_env);
  if (value == nullptr || value[0] == '\0') {
    return generation_stage_probe_selection::publication;
  }
  const std::string_view parsed{value};
  if (parsed == "off" || parsed == "0") {
    return generation_stage_probe_selection::off;
  }
  if (parsed == "publication") {
    return generation_stage_probe_selection::publication;
  }
  if (parsed == "selected") {
    return generation_stage_probe_selection::selected;
  }
  std::fprintf(stderr,
               "error: %s must be 'publication', 'selected', or 'off'\n",
               k_generation_stage_probe_env);
  std::exit(1);
}

bool should_capture_generation_stage_probe(
    const generation_fixture_spec &fixture_spec,
    const generation_case_spec &generation_case) {
  switch (generation_stage_probe_mode()) {
  case generation_stage_probe_selection::off:
    return false;
  case generation_stage_probe_selection::publication:
    return fixture_spec.fixture->current_publication &&
           generation_case.name == emel::bench::k_generation_case_name;
  case generation_stage_probe_selection::selected:
    return generation_workload_selected(generation_case);
  }
  return false;
}

std::filesystem::path bench_root_path() {
#ifdef EMEL_BENCH_REPO_ROOT
  return std::filesystem::path(EMEL_BENCH_REPO_ROOT);
#else
  std::filesystem::path path = std::filesystem::path(__FILE__).parent_path();
  path = path.parent_path().parent_path();
  return path;
#endif
}

std::string resolve_generation_model_path(const std::string_view fixture_rel) {
  return (bench_root_path() / fixture_rel).string();
}

std::filesystem::path generation_fixture_path(
    const emel::tools::generation_fixture_registry::maintained_fixture
        &fixture) {
  return bench_root_path() / fixture.fixture_rel;
}

bool generation_fixture_exists(
    const emel::tools::generation_fixture_registry::maintained_fixture
        &fixture) {
  return std::filesystem::exists(generation_fixture_path(fixture));
}

void report_missing_generation_fixture(
    const emel::tools::generation_fixture_registry::maintained_fixture
        &fixture) {
  std::fprintf(
      stderr, "warning: skipping missing generation fixture %.*s (%.*s)\n",
      static_cast<int>(fixture.name.size()), fixture.name.data(),
      static_cast<int>(fixture.fixture_rel.size()), fixture.fixture_rel.data());
}

extern std::string g_generation_formatter_contract;
extern std::string g_generation_architecture_contract;
extern std::string_view g_generation_fixture_rel;

[[noreturn]] void fail_bench_setup(const char *step, const char *detail) {
  const std::string_view fixture_rel = g_generation_fixture_rel.empty()
                                           ? std::string_view{"<unknown>"}
                                           : g_generation_fixture_rel;
  std::fprintf(stderr, "# generation_fixture: %.*s\n",
               static_cast<int>(fixture_rel.size()), fixture_rel.data());
  if (!g_generation_architecture_contract.empty()) {
    std::fprintf(stderr, "# generation_architecture: %.*s\n",
                 static_cast<int>(g_generation_architecture_contract.size()),
                 g_generation_architecture_contract.data());
  }
  if (!g_generation_formatter_contract.empty()) {
    std::fprintf(stderr, "# generation_formatter_contract: %.*s\n",
                 static_cast<int>(g_generation_formatter_contract.size()),
                 g_generation_formatter_contract.data());
  }
  std::fprintf(stderr, "error: generation bench setup failed at %s (%s)\n",
               step, detail);
  std::exit(1);
}

generation_case_spec
load_generation_case_spec(emel::bench::generation_workload_manifest manifest) {
  generation_case_spec spec = {};
  spec.manifest = std::move(manifest);
  spec.name = spec.manifest.case_name;
  spec.prompt = spec.manifest.prompt_text;
  spec.max_tokens = static_cast<int32_t>(spec.manifest.max_output_tokens);
  return spec;
}

const std::vector<generation_case_spec> &maintained_generation_variants() {
  static const std::vector<generation_case_spec> variants = [] {
    std::string error = {};
    std::vector<emel::bench::generation_workload_manifest> manifests = {};
    if (!emel::bench::load_generation_workload_manifests(bench_root_path(),
                                                         manifests, &error)) {
      fail_bench_setup("load_generation_workload_manifests", error.c_str());
    }
    std::vector<generation_case_spec> loaded = {};
    loaded.reserve(manifests.size());
    for (auto &manifest : manifests) {
      loaded.push_back(load_generation_case_spec(std::move(manifest)));
    }
    return loaded;
  }();
  return variants;
}

std::vector<generation_case_spec> generation_cases_for_fixture(
    const emel::tools::generation_fixture_registry::maintained_fixture &fixture,
    const bool comparable_only) {
  std::vector<generation_case_spec> cases = {};
  for (const generation_case_spec &candidate :
       maintained_generation_variants()) {
    if (candidate.manifest.fixture_rel != fixture.fixture_rel) {
      continue;
    }
    if (comparable_only && !candidate.manifest.comparable) {
      continue;
    }
    cases.push_back(candidate);
  }
  return cases;
}

void validate_generation_workload_fixture(
    const emel::tools::generation_fixture_registry::maintained_fixture &fixture,
    const generation_case_spec &spec) {
  if (spec.manifest.fixture_rel != fixture.fixture_rel ||
      spec.manifest.fixture_slug != fixture.slug ||
      spec.manifest.fixture_name != fixture.name) {
    fail_bench_setup("validate_generation_workload_fixture",
                     spec.manifest.id.c_str());
  }
}

void validate_generation_formatter_contract(
    const generation_case_spec &spec, const std::string_view actual_contract) {
  if (spec.manifest.formatter_contract != actual_contract) {
    fail_bench_setup("validate_generation_formatter_contract",
                     spec.manifest.id.c_str());
  }
}

struct llama_backend_guard {
  llama_backend_guard() { llama_backend_init(); }
  ~llama_backend_guard() { llama_backend_free(); }
};

void silence_llama_log(ggml_log_level, const char *, void *) {}

struct llama_log_silencer {
  ggml_log_callback callback = nullptr;
  void *user_data = nullptr;

  llama_log_silencer() {
    llama_log_get(&callback, &user_data);
    llama_log_set(silence_llama_log, nullptr);
  }

  ~llama_log_silencer() { llama_log_set(callback, user_data); }
};

void ensure_llama_backend_ready() {
  static llama_backend_guard backend_guard{};
  static llama_log_silencer log_silencer{};
  static_cast<void>(backend_guard);
  static_cast<void>(log_silencer);
}

template <size_t k_array_size>
void copy_name(std::array<char, k_array_size> &dst,
               const std::string_view value) {
  static_assert(k_array_size > 0, "copy_name requires non-empty destination");
  dst.fill('\0');
  const size_t copy_len = std::min(value.size(), k_array_size - 1);
  if (copy_len > 0u) {
    std::memcpy(dst.data(), value.data(), copy_len);
  }
}

emel::text::tokenizer::preprocessor::preprocessor_kind
generation_preprocessor_variant(const emel::model::data &model_data) {
  using preprocessor_kind =
      emel::text::tokenizer::preprocessor::preprocessor_kind;
  using tokenizer_model = emel::model::data::tokenizer_model;

  switch (model_data.vocab_data.tokenizer_model_id) {
  case tokenizer_model::SPM:
    return preprocessor_kind::spm;
  case tokenizer_model::BPE:
    return preprocessor_kind::bpe;
  case tokenizer_model::WPM:
    return preprocessor_kind::wpm;
  case tokenizer_model::UGM:
    return preprocessor_kind::ugm;
  case tokenizer_model::RWKV:
    return preprocessor_kind::rwkv;
  case tokenizer_model::PLAMO2:
    return preprocessor_kind::plamo2;
  case tokenizer_model::NONE:
  case tokenizer_model::UNKNOWN:
  default:
    return preprocessor_kind::fallback;
  }
}

emel::text::encoders::encoder_kind
generation_encoder_variant(const emel::model::data &model_data) {
  using encoder_kind = emel::text::encoders::encoder_kind;
  using tokenizer_model = emel::model::data::tokenizer_model;

  switch (model_data.vocab_data.tokenizer_model_id) {
  case tokenizer_model::SPM:
    return encoder_kind::spm;
  case tokenizer_model::BPE:
    return encoder_kind::bpe;
  case tokenizer_model::WPM:
    return encoder_kind::wpm;
  case tokenizer_model::UGM:
    return encoder_kind::ugm;
  case tokenizer_model::RWKV:
    return encoder_kind::rwkv;
  case tokenizer_model::PLAMO2:
    return encoder_kind::plamo2;
  case tokenizer_model::NONE:
  case tokenizer_model::UNKNOWN:
  default:
    return encoder_kind::fallback;
  }
}

std::string_view vocab_token_view(const emel::model::data::vocab &vocab,
                                  const int32_t token_id) {
  if (token_id < 0 || static_cast<uint32_t>(token_id) >= vocab.n_tokens) {
    return {};
  }

  const auto &entry = vocab.entries[static_cast<size_t>(token_id)];
  const size_t begin = static_cast<size_t>(entry.text_offset);
  const size_t length = static_cast<size_t>(entry.text_length);
  if (begin + length > static_cast<size_t>(vocab.token_bytes_used)) {
    return {};
  }

  return std::string_view{vocab.token_storage.data() + begin, length};
}

bool is_printable_ascii_token(const std::string_view piece) {
  if (piece.empty()) {
    return false;
  }

  for (const char ch : piece) {
    const uint8_t byte = static_cast<uint8_t>(ch);
    if (byte < 0x20u || byte > 0x7eu) {
      return false;
    }
  }
  return true;
}

struct gguf_capture {
  bool probe_done = false;
  bool probe_error = false;
  bool bind_done = false;
  bool bind_error = false;
  bool parse_done = false;
  bool parse_error = false;
  emel::gguf::loader::requirements requirements = {};
  emel::error::type err = emel::error::cast(emel::gguf::loader::error::none);
};

struct load_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
  uint64_t bytes_total = 0u;
  uint64_t bytes_done = 0u;
  bool used_mmap = false;
  emel::io::loader::event::strategy_kind requested_io_strategy =
      emel::io::loader::event::strategy_kind::none;
  emel::io::loader::event::strategy_kind used_io_strategy =
      emel::io::loader::event::strategy_kind::none;
};

struct initialize_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::text::generator::error::none);
};

struct generation_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::text::generator::error::none);
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

struct generation_result {
  std::array<char, k_generation_output_capacity> output = {};
  std::array<int32_t, k_generation_token_trace_capacity> output_token_ids = {};
  int32_t output_token_ids_count = 0;
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

void append_generation_token_id(generation_result &result,
                                const int32_t token_id) noexcept {
  const size_t index = static_cast<size_t>(result.output_token_ids_count);
  if (index >= result.output_token_ids.size()) {
    return;
  }

  result.output_token_ids[index] = token_id;
  result.output_token_ids_count += 1;
}

void capture_generation_output_metrics(emel::bench::result &record,
                                       const generation_result &generated) {
  const auto output_tokens =
      generated.tokens_generated > 0
          ? static_cast<std::uint64_t>(generated.tokens_generated)
          : 0u;
  record.output_tokens = output_tokens;
  record.tokens_per_second =
      emel::bench::compute_tokens_per_second(output_tokens, record.ns_per_op);
  record.output_bytes = static_cast<std::uint64_t>(generated.output_length);
  record.output_checksum = emel::bench::checksum_bytes(
      reinterpret_cast<const std::uint8_t *>(generated.output.data()),
      generated.output_length);
  record.output_text.assign(generated.output.data(), generated.output_length);
  record.output_token_ids_text.clear();
  for (int32_t idx = 0; idx < generated.output_token_ids_count; ++idx) {
    if (idx > 0) {
      record.output_token_ids_text.push_back(' ');
    }
    record.output_token_ids_text += std::to_string(generated.output_token_ids[idx]);
  }
}

struct prefill_probe_breakdown {
  std::uint64_t linear_ns = 0u;
  std::uint64_t attention_ns = 0u;
  std::uint64_t misc_ns = 0u;
  std::uint64_t misc_attention_norm_ns = 0u;
  std::uint64_t misc_qk_norm_ns = 0u;
  std::uint64_t misc_rope_ns = 0u;
  std::uint64_t misc_kv_store_ns = 0u;
  std::uint64_t misc_ctx_copy_ns = 0u;
  std::uint64_t misc_shortconv_ns = 0u;
  std::uint64_t shortconv_in_proj_ns = 0u;
  std::uint64_t shortconv_in_proj_prepare_ns = 0u;
  std::uint64_t shortconv_conv_ns = 0u;
  std::uint64_t shortconv_state_shift_ns = 0u;
  std::uint64_t shortconv_out_proj_ns = 0u;
  std::uint64_t shortconv_out_proj_prepare_ns = 0u;
  std::uint64_t misc_ffn_norm_ns = 0u;
  std::uint64_t misc_silu_ns = 0u;
};

struct emel_fixture {
  emel::model::data model_data = {};
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  uint64_t gguf_tensor_data_bytes = 0u;
  std::vector<uint8_t> read_copy_storage = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  uint32_t gguf_tensor_count = 0u;
  std::vector<emel::model::tensor::effect_request> effect_requests = {};
  std::vector<emel::model::tensor::effect_result> effect_results = {};
  std::vector<emel::io::event::tensor_load_span> io_load_spans = {};
  emel::gguf::loader::sm gguf_loader = {};
  emel::io::read::sm io_read = {};
  emel::io::staged_read::sm io_staged_read = {};
  emel::io::loader::sm io_loader{
      {.io_read = &io_read, .io_staged_read = &io_staged_read}};
  emel::model::tensor::sm tensor_loader = {};
  emel::model::loader::sm model_loader = {};
  gguf_capture gguf = {};
  load_capture load = {};
  emel::tools::generation_formatter_contract::formatter_binding
      formatter_binding = {};
};

struct reference_fixture {
  llama_model_ptr model = {nullptr, llama_model_free};
  llama_context_ptr context = {nullptr, llama_free};
  const llama_vocab *vocab = nullptr;
  int32_t vocab_size = 0;
  emel::tools::generation_formatter_contract::reference_formatter_info
      formatter = {};
};

struct generation_seam_audit {
  int32_t emel_reference_decode_calls = 0;
  int32_t emel_reference_logits_calls = 0;
  int32_t emel_reference_formatter_calls = 0;
  int32_t emel_reference_tokenize_calls = 0;
  int32_t emel_reference_vocab_calls = 0;
  int32_t direct_reference_decode_calls = 0;
  int32_t direct_reference_logits_calls = 0;
  int32_t direct_reference_formatter_calls = 0;
  int32_t direct_reference_tokenize_calls = 0;
  int32_t direct_reference_vocab_calls = 0;
};

struct generation_flash_evidence_state {
  bool ready = false;
  std::uint64_t flash_dispatch_calls = 0u;
  std::uint64_t optimized_flash_dispatch_calls = 0u;
  std::uint64_t shared_flash_dispatch_calls = 0u;
  std::uint32_t native_quantized_stage_count = 0u;
  std::uint32_t approved_dense_f32_stage_count = 0u;
  std::uint32_t disallowed_fallback_stage_count = 0u;
  std::uint32_t explicit_no_claim_stage_count = 0u;
  std::uint64_t native_q8_0_dispatch_calls = 0u;
  std::uint64_t packed_q8_0_dispatch_calls = 0u;
  std::uint64_t optimized_q2_dispatch_calls = 0u;
  std::uint64_t shared_q2_dispatch_calls = 0u;
  std::uint64_t optimized_q3_dispatch_calls = 0u;
  std::uint64_t shared_q3_dispatch_calls = 0u;
  std::uint64_t optimized_q4_dispatch_calls = 0u;
  std::uint64_t shared_q4_dispatch_calls = 0u;
  std::uint64_t optimized_q6_dispatch_calls = 0u;
  std::uint64_t shared_q6_dispatch_calls = 0u;
  generation_seam_audit seam = {};
};

struct emel_session {
  emel::model::data model_data = {};
  emel::model::generation::contract generation_contract = {};
  emel::text::tokenizer::sm tokenizer = {};
  emel::text::conditioner::sm conditioner = {};
  emel::kernel::matmul::lane_pool parallel_matmul_lanes = {};
  std::unique_ptr<emel::text::generator::sm> generator = {};
  std::array<emel::logits::sampler::fn, 1> samplers = {};
  emel::tools::generation_formatter_contract::formatter_binding
      formatter_binding = {};
  generation_seam_audit seam = {};
  initialize_capture initialize = {};
  generation_capture generation = {};
};

struct prepared_generation_fixture {
  const generation_fixture_spec *spec = nullptr;
  std::vector<generation_case_spec> cases = {};
  emel_fixture emel = {};
  reference_fixture reference = {};
};

struct prepared_emel_generation_fixture {
  const generation_fixture_spec *spec = nullptr;
  std::vector<generation_case_spec> cases = {};
  emel_fixture emel = {};
};

struct prepared_reference_generation_fixture {
  const generation_fixture_spec *spec = nullptr;
  std::vector<generation_case_spec> cases = {};
  reference_fixture reference = {};
};

std::string_view model_loader_error_name(const emel::error::type err) noexcept {
  switch (static_cast<emel::model::loader::error>(err)) {
  case emel::model::loader::error::none:
    return "none";
  case emel::model::loader::error::invalid_request:
    return "invalid_request";
  case emel::model::loader::error::parse_failed:
    return "parse_failed";
  case emel::model::loader::error::backend_error:
    return "backend_error";
  case emel::model::loader::error::model_invalid:
    return "model_invalid";
  case emel::model::loader::error::internal_error:
    return "internal_error";
  case emel::model::loader::error::io_strategy_unavailable:
    return "io_strategy_unavailable";
  case emel::model::loader::error::untracked:
    return "untracked";
  }
  return "unknown";
}

emel::model::detail::kv_binding
kv_binding_from_fixture(const emel_fixture &fixture) {
  return emel::model::detail::kv_binding{
      .arena = std::span<const uint8_t>{fixture.kv_arena.data(),
                                        fixture.kv_arena.size()},
      .entries =
          std::span<const emel::gguf::loader::kv_entry>{
              fixture.kv_entries.data(), fixture.kv_entries.size()},
  };
}

generation_flash_evidence_state g_generation_flash_evidence = {};
std::vector<emel::bench::generation_stage_probe> g_generation_stage_probes = {};
std::string g_generation_formatter_contract = {};
std::string g_generation_architecture_contract = {};
std::string_view g_generation_fixture_rel = {};
emel::bench::generation_lane_mode g_generation_lane_mode =
    emel::bench::generation_lane_mode::emel;

uint32_t read_u32_le(const std::span<const uint8_t> bytes) {
  uint32_t value = 0u;
  for (size_t i = 0u; i < sizeof(uint32_t); ++i) {
    value |= static_cast<uint32_t>(bytes[i]) << (i * 8u);
  }
  return value;
}

uint64_t read_u64_le(const std::span<const uint8_t> bytes) {
  uint64_t value = 0u;
  for (size_t i = 0u; i < sizeof(uint64_t); ++i) {
    value |= static_cast<uint64_t>(bytes[i]) << (i * 8u);
  }
  return value;
}

emel::error::type sampler_select_argmax(int32_t &candidate_ids,
                                        float &candidate_scores,
                                        int32_t &candidate_count,
                                        int32_t &selected_token_out) {
  int32_t best_index = 0;
  float best_score = (&candidate_scores)[0];
  for (int32_t idx = 1; idx < candidate_count; ++idx) {
    if ((&candidate_scores)[idx] > best_score) {
      best_score = (&candidate_scores)[idx];
      best_index = idx;
    }
  }

  selected_token_out = (&candidate_ids)[best_index];
  return emel::error::cast(emel::logits::sampler::error::none);
}

template <class fixture_type> void reset_gguf_capture(fixture_type &fixture) {
  fixture.gguf = {};
}

void reset_load_capture(emel_fixture &fixture) { fixture.load = {}; }
void reset_initialize_capture(emel_session &session) {
  session.initialize = {};
}
void reset_generation_capture(emel_session &session) {
  session.generation = {};
}

bool capture_generator_diagnostics(
    emel_session &session,
    emel::text::generator::diagnostics &diagnostics_out) {
  if (session.generator == nullptr) {
    return false;
  }
  return session.generator->process_event(
      emel::text::generator::event::capture_diagnostics{diagnostics_out});
}

bool configure_generator_benchmark_lane(emel_session &session) {
  if (session.generator == nullptr) {
    return false;
  }
  emel::text::generator::event::configure_benchmark_lane ev{};
  ev.lane = generator_benchmark_lane_event_value();
  return session.generator->process_event(ev);
}

template <class fixture_type>
void on_probe_done_impl(void *owner,
                        const emel::gguf::loader::events::probe_done &ev) {
  auto &fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.probe_done = true;
  fixture.gguf.probe_error = false;
  fixture.gguf.requirements = ev.requirements_out;
}

void on_probe_done(void *owner,
                   const emel::gguf::loader::events::probe_done &ev) {
  on_probe_done_impl<emel_fixture>(owner, ev);
}

template <class fixture_type>
void on_probe_error_impl(void *owner,
                         const emel::gguf::loader::events::probe_error &ev) {
  auto &fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.probe_error = true;
  fixture.gguf.err = ev.err;
}

void on_probe_error(void *owner,
                    const emel::gguf::loader::events::probe_error &ev) {
  on_probe_error_impl<emel_fixture>(owner, ev);
}

template <class fixture_type>
void on_bind_done_impl(void *owner,
                       const emel::gguf::loader::events::bind_done &) {
  auto &fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.bind_done = true;
  fixture.gguf.bind_error = false;
}

void on_bind_done(void *owner,
                  const emel::gguf::loader::events::bind_done &ev) {
  on_bind_done_impl<emel_fixture>(owner, ev);
}

template <class fixture_type>
void on_bind_error_impl(void *owner,
                        const emel::gguf::loader::events::bind_error &ev) {
  auto &fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.bind_error = true;
  fixture.gguf.err = ev.err;
}

void on_bind_error(void *owner,
                   const emel::gguf::loader::events::bind_error &ev) {
  on_bind_error_impl<emel_fixture>(owner, ev);
}

template <class fixture_type>
void on_parse_done_impl(void *owner,
                        const emel::gguf::loader::events::parse_done &) {
  auto &fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.parse_done = true;
  fixture.gguf.parse_error = false;
}

void on_parse_done(void *owner,
                   const emel::gguf::loader::events::parse_done &ev) {
  on_parse_done_impl<emel_fixture>(owner, ev);
}

template <class fixture_type>
void on_parse_error_impl(void *owner,
                         const emel::gguf::loader::events::parse_error &ev) {
  auto &fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.parse_error = true;
  fixture.gguf.err = ev.err;
}

void on_parse_error(void *owner,
                    const emel::gguf::loader::events::parse_error &ev) {
  on_parse_error_impl<emel_fixture>(owner, ev);
}

void on_load_done(void *owner,
                  const emel::model::loader::events::load_done &ev) {
  auto &fixture = *static_cast<emel_fixture *>(owner);
  fixture.load.done = true;
  fixture.load.error = false;
  fixture.load.err = emel::error::cast(emel::model::loader::error::none);
  fixture.load.bytes_total = ev.bytes_total;
  fixture.load.bytes_done = ev.bytes_done;
  fixture.load.used_mmap = ev.used_mmap;
  fixture.load.used_io_strategy = ev.used_io_strategy;
}

void on_load_error(void *owner,
                   const emel::model::loader::events::load_error &ev) {
  auto &fixture = *static_cast<emel_fixture *>(owner);
  fixture.load.error = true;
  fixture.load.err = ev.err;
  fixture.load.requested_io_strategy = ev.requested_io_strategy;
  fixture.load.used_io_strategy = ev.used_io_strategy;
}

void on_initialize_done(
    void *owner, const emel::text::generator::events::initialize_done &) {
  auto &session = *static_cast<emel_session *>(owner);
  session.initialize.done = true;
  session.initialize.error = false;
  session.initialize.err =
      emel::error::cast(emel::text::generator::error::none);
}

void on_initialize_error(
    void *owner, const emel::text::generator::events::initialize_error &ev) {
  auto &session = *static_cast<emel_session *>(owner);
  session.initialize.error = true;
  session.initialize.err = ev.err;
}

void on_generation_done(
    void *owner, const emel::text::generator::events::generation_done &ev) {
  auto &session = *static_cast<emel_session *>(owner);
  session.generation.done = true;
  session.generation.error = false;
  session.generation.err =
      emel::error::cast(emel::text::generator::error::none);
  session.generation.tokens_generated = ev.tokens_generated;
  session.generation.output_length = ev.output_length;
}

void on_generation_error(
    void *owner, const emel::text::generator::events::generation_error &ev) {
  auto &session = *static_cast<emel_session *>(owner);
  session.generation.error = true;
  session.generation.err = ev.err;
  session.generation.tokens_generated = ev.tokens_generated;
  session.generation.output_length = ev.output_length;
}

bool tokenizer_bind_dispatch(void *tokenizer_sm,
                             const emel::text::tokenizer::event::bind &ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)
      ->process_event(ev);
}

bool tokenizer_tokenize_dispatch(
    void *tokenizer_sm, const emel::text::tokenizer::event::tokenize &ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)
      ->process_event(ev);
}

void reset_generation_seam(generation_seam_audit &seam) { seam = {}; }

void reset_generation_flash_evidence() {
  g_generation_flash_evidence = {};
  g_generation_stage_probes.clear();
  g_generation_formatter_contract.clear();
  g_generation_architecture_contract.clear();
  g_generation_fixture_rel = {};
}

bool generation_seam_audit_enabled() {
  return env_enabled("EMEL_BENCH_AUDIT_GENERATION_SEAMS");
}

void print_generation_seam_audit(const char *label,
                                 const generation_seam_audit &seam) {
  std::fprintf(
      stderr,
      "generation_bench_seams[%s]: emel_decode_calls=%d emel_logits_calls=%d "
      "emel_formatter_calls=%d emel_tokenize_calls=%d emel_vocab_calls=%d "
      "reference_decode_calls=%d reference_logits_calls=%d "
      "reference_formatter_calls=%d reference_tokenize_calls=%d "
      "reference_vocab_calls=%d\n",
      label, seam.emel_reference_decode_calls, seam.emel_reference_logits_calls,
      seam.emel_reference_formatter_calls, seam.emel_reference_tokenize_calls,
      seam.emel_reference_vocab_calls, seam.direct_reference_decode_calls,
      seam.direct_reference_logits_calls, seam.direct_reference_formatter_calls,
      seam.direct_reference_tokenize_calls, seam.direct_reference_vocab_calls);
}

void verify_emel_generation_seam(const generation_seam_audit &seam) {
  if (seam.emel_reference_decode_calls != 0 ||
      seam.emel_reference_logits_calls != 0 ||
      seam.emel_reference_formatter_calls != 0 ||
      seam.emel_reference_tokenize_calls != 0 ||
      seam.emel_reference_vocab_calls != 0 ||
      seam.direct_reference_decode_calls != 0 ||
      seam.direct_reference_logits_calls != 0 ||
      seam.direct_reference_formatter_calls != 0 ||
      seam.direct_reference_tokenize_calls != 0 ||
      seam.direct_reference_vocab_calls != 0) {
    fail_bench_setup("generation seam audit",
                     "EMEL benchmark path touched reference decode seam");
  }
}

std::string_view generator_error_name(const emel::error::type err) noexcept {
  switch (static_cast<emel::text::generator::error>(err)) {
  case emel::text::generator::error::none:
    return "none";
  case emel::text::generator::error::invalid_request:
    return "invalid_request";
  case emel::text::generator::error::backend:
    return "backend";
  }
  return "unknown";
}

void verify_reference_generation_seam(const generation_seam_audit &seam) {
  if (seam.emel_reference_decode_calls != 0 ||
      seam.emel_reference_logits_calls != 0 ||
      seam.emel_reference_formatter_calls != 0 ||
      seam.emel_reference_tokenize_calls != 0 ||
      seam.emel_reference_vocab_calls != 0 ||
      seam.direct_reference_decode_calls <= 0 ||
      seam.direct_reference_logits_calls <= 0 ||
      seam.direct_reference_formatter_calls <= 0 ||
      seam.direct_reference_tokenize_calls <= 0 ||
      seam.direct_reference_vocab_calls <= 0) {
    fail_bench_setup(
        "generation seam audit",
        "reference benchmark path did not stay on the explicit reference seam");
  }
}

int32_t run_direct_reference_decode(generation_seam_audit &seam,
                                    llama_context *ctx,
                                    const llama_batch batch) {
  seam.direct_reference_decode_calls += 1;
  return llama_decode(ctx, batch);
}

float *read_direct_reference_logits(generation_seam_audit &seam,
                                    llama_context *ctx) {
  seam.direct_reference_logits_calls += 1;
  return llama_get_logits_ith(ctx, -1);
}

bool format_direct_reference_prompt(
    generation_seam_audit &seam,
    const emel::tools::generation_formatter_contract::reference_formatter_info
        &formatter,
    const std::string_view prompt, std::string &formatted_prompt) {
  seam.direct_reference_formatter_calls += 1;
  return emel::tools::generation_formatter_contract::
      format_reference_single_user_prompt(formatter, prompt, formatted_prompt);
}

int32_t tokenize_direct_reference_prompt(generation_seam_audit &seam,
                                         const llama_vocab *vocab,
                                         const std::string &formatted_prompt,
                                         llama_token *tokens,
                                         const int32_t token_capacity) {
  seam.direct_reference_tokenize_calls += 1;
  return llama_tokenize(vocab, formatted_prompt.data(),
                        static_cast<int32_t>(formatted_prompt.size()), tokens,
                        token_capacity, false, false);
}

bool reference_vocab_is_control(generation_seam_audit &seam,
                                const llama_vocab *vocab,
                                const llama_token token) {
  seam.direct_reference_vocab_calls += 1;
  return llama_vocab_is_control(vocab, token);
}

bool reference_vocab_is_eog(generation_seam_audit &seam,
                            const llama_vocab *vocab, const llama_token token) {
  seam.direct_reference_vocab_calls += 1;
  return llama_vocab_is_eog(vocab, token);
}

const char *reference_vocab_text(generation_seam_audit &seam,
                                 const llama_vocab *vocab,
                                 const llama_token token) {
  seam.direct_reference_vocab_calls += 1;
  return llama_vocab_get_text(vocab, token);
}

emel::error::type map_gguf_error(const emel::error::type err) {
  using gguf_error = emel::gguf::loader::error;
  using model_error = emel::model::loader::error;

  switch (err) {
  case emel::error::cast(gguf_error::none):
    return emel::error::cast(model_error::none);
  case emel::error::cast(gguf_error::invalid_request):
    return emel::error::cast(model_error::invalid_request);
  case emel::error::cast(gguf_error::model_invalid):
    return emel::error::cast(model_error::model_invalid);
  case emel::error::cast(gguf_error::capacity):
    return emel::error::cast(model_error::backend_error);
  case emel::error::cast(gguf_error::parse_failed):
    return emel::error::cast(model_error::parse_failed);
  case emel::error::cast(gguf_error::internal_error):
    return emel::error::cast(model_error::internal_error);
  case emel::error::cast(gguf_error::untracked):
  default:
    return emel::error::cast(model_error::untracked);
  }
}

template <class fixture_type>
std::string_view kv_key_view(const fixture_type &fixture,
                             const emel::gguf::loader::kv_entry &entry) {
  if (static_cast<size_t>(entry.key_offset) +
          static_cast<size_t>(entry.key_length) >
      fixture.kv_arena.size()) {
    return {};
  }

  return std::string_view{
      reinterpret_cast<const char *>(fixture.kv_arena.data() +
                                     entry.key_offset),
      entry.key_length,
  };
}

template <class fixture_type>
std::span<const uint8_t>
kv_value_view(const fixture_type &fixture,
              const emel::gguf::loader::kv_entry &entry) {
  if (static_cast<size_t>(entry.value_offset) +
          static_cast<size_t>(entry.value_length) >
      fixture.kv_arena.size()) {
    return {};
  }

  return std::span<const uint8_t>{fixture.kv_arena.data() + entry.value_offset,
                                  entry.value_length};
}

template <class fixture_type>
const emel::gguf::loader::kv_entry *find_kv_entry(const fixture_type &fixture,
                                                  const std::string_view key) {
  for (const auto &entry : fixture.kv_entries) {
    if (kv_key_view(fixture, entry) == key) {
      return &entry;
    }
  }
  return nullptr;
}

template <class fixture_type>
bool decode_string_value(const fixture_type &fixture,
                         const emel::gguf::loader::kv_entry &entry,
                         std::string_view &value_out);

template <class fixture_type>
emel::tools::generation_formatter_contract::formatter_binding
resolve_fixture_formatter_binding(const fixture_type &fixture) {
  std::string_view primary_template = {};
  const auto *entry = find_kv_entry(fixture, "tokenizer.chat_template");
  if (entry != nullptr &&
      !decode_string_value(fixture, *entry, primary_template)) {
    return emel::tools::generation_formatter_contract::formatter_binding{
        .formatter_ctx = nullptr,
        .format_prompt = emel::text::formatter::format_raw,
        .support = emel::tools::generation_formatter_contract::support_kind::
            unsupported_template,
        .contract = emel::tools::generation_formatter_contract::
            k_unsupported_template_contract,
    };
  }

  uint32_t named_template_count = 0u;
  for (const auto &candidate : fixture.kv_entries) {
    const std::string_view key = kv_key_view(fixture, candidate);
    if (key.starts_with("tokenizer.chat_template.") &&
        key != "tokenizer.chat_template") {
      named_template_count += 1u;
    }
  }

  return emel::tools::generation_formatter_contract::
      resolve_primary_template_binding(primary_template, named_template_count);
}

template <class fixture_type>
bool decode_integer_value(const fixture_type &fixture,
                          const emel::gguf::loader::kv_entry &entry,
                          uint64_t &value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::constants;

  switch (entry.value_type) {
  case constants::gguf_type_uint8:
    if (bytes.size() != 1u) {
      return false;
    }
    value_out = bytes[0];
    return true;
  case constants::gguf_type_int8:
    if (bytes.size() != 1u) {
      return false;
    }
    value_out = static_cast<uint64_t>(static_cast<int8_t>(bytes[0]));
    return true;
  case constants::gguf_type_uint16:
  case constants::gguf_type_int16:
    if (bytes.size() != 2u) {
      return false;
    }
    value_out = static_cast<uint64_t>(bytes[0]) |
                (static_cast<uint64_t>(bytes[1]) << 8u);
    return true;
  case constants::gguf_type_uint32:
  case constants::gguf_type_int32:
    if (bytes.size() != sizeof(uint32_t)) {
      return false;
    }
    value_out = read_u32_le(bytes);
    return true;
  case constants::gguf_type_uint64:
  case constants::gguf_type_int64:
    if (bytes.size() != sizeof(uint64_t)) {
      return false;
    }
    value_out = read_u64_le(bytes);
    return true;
  default:
    return false;
  }
}

template <class fixture_type>
bool decode_float_value(const fixture_type &fixture,
                        const emel::gguf::loader::kv_entry &entry,
                        float &value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::constants;

  if (entry.value_type == constants::gguf_type_float32) {
    if (bytes.size() != sizeof(float)) {
      return false;
    }
    std::memcpy(&value_out, bytes.data(), sizeof(float));
    return true;
  }

  if (entry.value_type == constants::gguf_type_float64) {
    if (bytes.size() != sizeof(double)) {
      return false;
    }
    double value = 0.0;
    std::memcpy(&value, bytes.data(), sizeof(double));
    value_out = static_cast<float>(value);
    return true;
  }

  return false;
}

template <class fixture_type>
bool decode_bool_value(const fixture_type &fixture,
                       const emel::gguf::loader::kv_entry &entry,
                       bool &value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::constants;
  if (entry.value_type != constants::gguf_type_bool || bytes.size() != 1u) {
    return false;
  }

  value_out = bytes[0] != 0u;
  return true;
}

template <class fixture_type>
bool decode_string_value(const fixture_type &fixture,
                         const emel::gguf::loader::kv_entry &entry,
                         std::string_view &value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::constants;

  if (entry.value_type != constants::gguf_type_string ||
      bytes.size() < sizeof(uint64_t)) {
    return false;
  }

  const uint64_t length = read_u64_le(bytes.first(sizeof(uint64_t)));
  if (length > bytes.size() - sizeof(uint64_t)) {
    return false;
  }

  value_out = std::string_view{
      reinterpret_cast<const char *>(bytes.data() + sizeof(uint64_t)),
      static_cast<size_t>(length),
  };
  return true;
}

template <class fixture_type>
bool decode_string_array_count(const fixture_type &fixture,
                               const emel::gguf::loader::kv_entry &entry,
                               uint32_t &count_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::constants;

  if (entry.value_type != constants::gguf_type_array ||
      bytes.size() < sizeof(uint32_t) + sizeof(uint64_t)) {
    return false;
  }

  const uint32_t element_type = read_u32_le(bytes.first(sizeof(uint32_t)));
  if (element_type != constants::gguf_type_string) {
    return false;
  }

  const uint64_t count =
      read_u64_le(bytes.subspan(sizeof(uint32_t), sizeof(uint64_t)));
  if (count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  count_out = static_cast<uint32_t>(count);
  return true;
}

template <class fixture_type>
bool decode_integer_array_first_nonzero(
    const fixture_type &fixture, const emel::gguf::loader::kv_entry &entry,
    int32_t &value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::constants;

  if (entry.value_type != constants::gguf_type_array ||
      bytes.size() < sizeof(uint32_t) + sizeof(uint64_t)) {
    return false;
  }

  const uint32_t element_type = read_u32_le(bytes.first(sizeof(uint32_t)));
  const uint64_t count =
      read_u64_le(bytes.subspan(sizeof(uint32_t), sizeof(uint64_t)));
  const std::span<const uint8_t> payload =
      bytes.subspan(sizeof(uint32_t) + sizeof(uint64_t));

  size_t element_size = 0u;
  switch (element_type) {
  case constants::gguf_type_uint8:
  case constants::gguf_type_int8:
    element_size = 1u;
    break;
  case constants::gguf_type_uint16:
  case constants::gguf_type_int16:
    element_size = 2u;
    break;
  case constants::gguf_type_uint32:
  case constants::gguf_type_int32:
    element_size = 4u;
    break;
  case constants::gguf_type_uint64:
  case constants::gguf_type_int64:
    element_size = 8u;
    break;
  default:
    return false;
  }

  if (count == 0u || payload.size() != count * element_size) {
    return false;
  }

  for (uint64_t index = 0u; index < count; ++index) {
    const std::span<const uint8_t> element = payload.subspan(
        static_cast<size_t>(index * element_size), element_size);
    uint64_t raw_value = 0u;
    switch (element_type) {
    case constants::gguf_type_uint8:
      raw_value = element[0];
      break;
    case constants::gguf_type_int8:
      raw_value = static_cast<uint64_t>(static_cast<int8_t>(element[0]));
      break;
    case constants::gguf_type_uint16:
    case constants::gguf_type_int16:
      raw_value = static_cast<uint64_t>(element[0]) |
                  (static_cast<uint64_t>(element[1]) << 8u);
      break;
    case constants::gguf_type_uint32:
    case constants::gguf_type_int32:
      raw_value = read_u32_le(element);
      break;
    case constants::gguf_type_uint64:
    case constants::gguf_type_int64:
      raw_value = read_u64_le(element);
      break;
    default:
      return false;
    }

    if (raw_value == 0u ||
        raw_value >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      continue;
    }

    value_out = static_cast<int32_t>(raw_value);
    return true;
  }

  return false;
}

bool copy_tensor_names(const std::span<const uint8_t> file_image,
                       emel::model::data &model_data) {
  model_data.name_bytes_used = 0u;

  for (uint32_t i = 0u; i < model_data.n_tensors; ++i) {
    auto &tensor = model_data.tensors[i];
    const size_t name_offset = static_cast<size_t>(tensor.name_offset);
    const size_t name_length = static_cast<size_t>(tensor.name_length);
    if (name_offset + name_length > file_image.size() ||
        model_data.name_bytes_used + name_length >
            model_data.name_storage.size()) {
      return false;
    }

    const uint32_t copied_offset = model_data.name_bytes_used;
    if (name_length > 0u) {
      std::memcpy(model_data.name_storage.data() + copied_offset,
                  file_image.data() + name_offset, name_length);
    }

    model_data.name_bytes_used += static_cast<uint32_t>(name_length);
    tensor.name_offset = copied_offset;
  }
  return true;
}

std::string_view
tensor_name_view(const emel::model::data &model_data,
                 const emel::model::data::tensor_record &tensor) {
  if (static_cast<size_t>(tensor.name_offset) +
          static_cast<size_t>(tensor.name_length) >
      model_data.name_storage.size()) {
    return {};
  }

  return std::string_view{model_data.name_storage.data() + tensor.name_offset,
                          tensor.name_length};
}

bool try_parse_block_index(const std::string_view name,
                           int32_t &block_index_out) {
  constexpr std::string_view k_prefix = "blk.";
  if (!name.starts_with(k_prefix)) {
    return false;
  }

  size_t cursor = k_prefix.size();
  if (cursor >= name.size()) {
    return false;
  }

  int32_t value = 0;
  bool saw_digit = false;
  while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
    saw_digit = true;
    value = value * 10 + static_cast<int32_t>(name[cursor] - '0');
    ++cursor;
  }

  if (!saw_digit || cursor >= name.size() || name[cursor] != '.') {
    return false;
  }

  block_index_out = value;
  return true;
}

emel::error::type populate_model_metadata(const emel_fixture &fixture,
                                          emel::model::data &model_data) {
  return emel::model::detail::load_hparams_from_gguf(
             kv_binding_from_fixture(fixture), model_data)
             ? emel::error::cast(emel::model::loader::error::none)
             : emel::error::cast(emel::model::loader::error::model_invalid);
}

emel::error::type prebind_emel_gguf_storage(emel_fixture &fixture) {
  if (fixture.file_bytes.empty()) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }

  const std::span<const uint8_t> file_image{
      fixture.file_bytes.data(),
      fixture.file_bytes.size(),
  };

  fixture.gguf_tensor_count = 0u;
  reset_gguf_capture(fixture);
  emel::gguf::loader::requirements requirements = {};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb{&fixture,
                                                               on_probe_done};
  const emel::gguf::loader::event::probe_error_fn probe_error_cb{
      &fixture, on_probe_error};
  const emel::gguf::loader::event::probe probe_ev{
      file_image,
      requirements,
      probe_done_cb,
      probe_error_cb,
  };
  if (!fixture.gguf_loader.process_event(probe_ev) ||
      !fixture.gguf.probe_done || fixture.gguf.probe_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  if (requirements.tensor_count >
      static_cast<uint32_t>(emel::model::data::k_max_tensors)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const uint64_t arena_bytes =
      emel::gguf::loader::required_kv_arena_bytes(requirements);
  if (arena_bytes == std::numeric_limits<uint64_t>::max()) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }

  fixture.kv_arena.resize(static_cast<size_t>(arena_bytes));
  fixture.kv_entries.resize(requirements.kv_count);
  fixture.gguf_tensor_count = requirements.tensor_count;
  fixture.gguf_tensor_data_bytes = requirements.tensor_data_bytes;
  return emel::error::cast(emel::model::loader::error::none);
}

std::string_view architecture_name_view(const emel::model::data &model_data) {
  size_t length = 0u;
  while (length < model_data.architecture_name.size() &&
         model_data.architecture_name[length] != '\0') {
    ++length;
  }
  return std::string_view{model_data.architecture_name.data(), length};
}

emel::error::type
run_emel_parse_model(void *owner, const emel::model::loader::event::load &req) {
  auto &fixture = *static_cast<emel_fixture *>(owner);
  if (req.file_image == nullptr || req.file_size == 0u) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }

  const std::span<const uint8_t> file_image{
      static_cast<const uint8_t *>(req.file_image),
      static_cast<size_t>(req.file_size),
  };

  reset_gguf_capture(fixture);
  const emel::gguf::loader::event::bind_done_fn bind_done_cb{&fixture,
                                                             on_bind_done};
  const emel::gguf::loader::event::bind_error_fn bind_error_cb{&fixture,
                                                               on_bind_error};
  const emel::gguf::loader::event::bind_storage bind_ev{
      std::span<uint8_t>{fixture.kv_arena},
      std::span<emel::gguf::loader::kv_entry>{fixture.kv_entries},
      std::span<emel::model::data::tensor_record>{req.model_data.tensors.data(),
                                                  fixture.gguf_tensor_count},
      bind_done_cb,
      bind_error_cb,
  };
  if (!fixture.gguf_loader.process_event(bind_ev) || !fixture.gguf.bind_done ||
      fixture.gguf.bind_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  reset_gguf_capture(fixture);
  const emel::gguf::loader::event::parse_done_fn parse_done_cb{&fixture,
                                                               on_parse_done};
  const emel::gguf::loader::event::parse_error_fn parse_error_cb{
      &fixture, on_parse_error};
  const emel::gguf::loader::event::parse parse_ev{
      file_image,
      parse_done_cb,
      parse_error_cb,
  };
  if (!fixture.gguf_loader.process_event(parse_ev) ||
      !fixture.gguf.parse_done || fixture.gguf.parse_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  req.model_data.n_tensors = fixture.gguf_tensor_count;
  if (!copy_tensor_names(file_image, req.model_data)) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }

  return populate_model_metadata(fixture, req.model_data);
}

emel::error::type
run_emel_map_layers(void *, const emel::model::loader::event::load &req) {
  int32_t max_block_index = -1;
  for (uint32_t i = 0u; i < req.model_data.n_tensors; ++i) {
    int32_t block_index = -1;
    if (emel::model::try_parse_block_index(
            emel::model::tensor_name_view(req.model_data,
                                          req.model_data.tensors[i]),
            block_index) &&
        block_index > max_block_index) {
      max_block_index = block_index;
    }
  }

  if (max_block_index >= 0) {
    req.model_data.n_layers = max_block_index + 1;
    return emel::error::cast(emel::model::loader::error::none);
  }
  if (req.model_data.params.n_layer > 0) {
    req.model_data.n_layers = req.model_data.params.n_layer;
    return emel::error::cast(emel::model::loader::error::none);
  }
  return emel::error::cast(emel::model::loader::error::model_invalid);
}

emel::error::type
run_emel_validate_structure(void *,
                            const emel::model::loader::event::load &req) {
  if (req.model_data.n_tensors == 0u || req.model_data.n_layers <= 0 ||
      req.model_data.weights_data == nullptr ||
      req.model_data.weights_size == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
run_emel_validate_architecture(void *,
                               const emel::model::loader::event::load &req) {
  return emel::model::validate_execution_contract(req.model_data);
}

bool prepare_emel_fixture(emel_fixture &fixture,
                          const std::string &model_path) {
  if (emel::io::source::load_file_bytes(model_path, fixture.file_bytes) !=
      emel::error::cast(emel::io::read::error::none)) {
    return false;
  }

  if (prebind_emel_gguf_storage(fixture) !=
      emel::error::cast(emel::model::loader::error::none)) {
    return false;
  }

  reset_load_capture(fixture);
  fixture.effect_requests.resize(emel::model::data::k_max_tensors);
  fixture.effect_results.resize(emel::model::data::k_max_tensors);
  fixture.io_load_spans.resize(emel::model::data::k_max_tensors);
  emel::model::loader::event::parse_model_fn parse_model{&fixture,
                                                         run_emel_parse_model};
  emel::model::loader::event::load load_ev{fixture.model_data, parse_model};
  load_ev.model_path = model_path;
  load_ev.file_image = fixture.file_bytes.data();
  load_ev.file_size = fixture.file_bytes.size();
  load_ev.tensor_loader = &fixture.tensor_loader;
  load_ev.effect_requests = std::span{fixture.effect_requests};
  load_ev.effect_results = std::span{fixture.effect_results};
  load_ev.io_load_spans = std::span<emel::io::event::tensor_load_span>{
      fixture.io_load_spans.data(), fixture.io_load_spans.size()};
  emel::tools::bind_model_load_io_strategy(load_ev, fixture.io_loader);
  if (load_ev.io_strategy ==
          emel::io::loader::event::strategy_kind::read_copy ||
      load_ev.io_strategy ==
          emel::io::loader::event::strategy_kind::staged_read) {
    fixture.read_copy_storage.resize(
        static_cast<size_t>(fixture.gguf_tensor_data_bytes));
    load_ev.read_copy_storage = std::span<uint8_t>{fixture.read_copy_storage};
  }
  load_ev.map_layers = {nullptr, run_emel_map_layers};
  load_ev.validate_structure = {nullptr, run_emel_validate_structure};
  load_ev.validate_architecture_impl = {nullptr,
                                        run_emel_validate_architecture};
  load_ev.on_done = {&fixture, on_load_done};
  load_ev.on_error = {&fixture, on_load_error};
  if (!fixture.model_loader.process_event(load_ev) || !fixture.load.done ||
      fixture.load.error) {
    fixture.formatter_binding = resolve_fixture_formatter_binding(fixture);
    return false;
  }
  fixture.formatter_binding = resolve_fixture_formatter_binding(fixture);
  if (!emel::tools::generation_formatter_contract::binding_supported(
          fixture.formatter_binding)) {
    return false;
  }
  return true;
}

llama_model_ptr load_reference_model(const std::string &model_path);
llama_context_ptr
make_reference_context(llama_model *model,
                       ggml_backend_sched_eval_callback eval_callback = nullptr,
                       void *eval_user_data = nullptr);
bool run_reference_generate_preloaded(
    const reference_fixture &fixture, const generation_case_spec &spec,
    llama_context *ctx, const std::vector<llama_token> &prompt_tokens,
    generation_seam_audit &seam, generation_result &result_out);

bool prepare_reference_fixture(reference_fixture &fixture,
                               const std::string &model_path) {
  fixture.model = load_reference_model(model_path);
  if (fixture.model == nullptr) {
    return false;
  }

  fixture.vocab = llama_model_get_vocab(fixture.model.get());
  if (fixture.vocab == nullptr) {
    return false;
  }

  fixture.formatter = emel::tools::generation_formatter_contract::
      resolve_reference_formatter_info(fixture.model.get());
  if (!emel::tools::generation_formatter_contract::reference_binding_supported(
          fixture.formatter)) {
    return false;
  }

  fixture.vocab_size = llama_vocab_n_tokens(fixture.vocab);
  if (fixture.vocab_size <= 0) {
    return false;
  }

  fixture.context = make_reference_context(fixture.model.get());
  return fixture.context != nullptr;
}

llama_model_ptr load_reference_model(const std::string &model_path) {
  llama_model_params params = llama_model_default_params();
  // Force the reference path onto CPU so the compare stays aligned with EMEL's
  // CPU-only backend.
  params.n_gpu_layers = 0;
  params.check_tensors = false;
  return llama_model_ptr{llama_model_load_from_file(model_path.c_str(), params),
                         llama_model_free};
}

bool reference_graph_contains_flash_attn_op(llama_context &ctx) {
  const auto &cparams = ctx.get_cparams();
  if (!cparams.flash_attn || cparams.auto_fa) {
    return false;
  }

  llama_memory_context_ptr mctx;
  if (llama_memory_t memory = ctx.get_memory()) {
    mctx = memory->init_full();
  }

  ggml_cgraph *graph = ctx.graph_reserve(1u, 1u, 1u, mctx.get(), true);
  if (graph == nullptr) {
    return false;
  }

  for (int32_t idx = 0; idx < ggml_graph_n_nodes(graph); ++idx) {
    ggml_tensor *node = ggml_graph_node(graph, idx);
    if (node != nullptr && node->op == GGML_OP_FLASH_ATTN_EXT) {
      return true;
    }
  }
  return false;
}

llama_context_ptr init_reference_context(llama_model *model,
                                         const llama_context_params &params) {
  return llama_context_ptr{
      model != nullptr ? llama_init_from_model(model, params) : nullptr,
      llama_free};
}

llama_context_ptr
make_reference_context(llama_model *model,
                       ggml_backend_sched_eval_callback eval_callback,
                       void *eval_user_data) {
  llama_context_params params = llama_context_default_params();
  params.flash_attn_type = k_reference_flash_attn_type;
  params.n_ctx = 0;
  params.n_batch = 512;
  params.n_ubatch = 512;
  params.n_seq_max = 1;
  params.n_threads = generation_reference_decode_thread_count();
  params.n_threads_batch = generation_reference_batch_thread_count();
  params.embeddings = false;
  params.cb_eval = eval_callback;
  params.cb_eval_user_data = eval_user_data;
  llama_context_ptr probe = init_reference_context(model, params);
  if (probe != nullptr && !reference_graph_contains_flash_attn_op(*probe)) {
    fail_bench_setup("make_reference_context",
                     "reference graph missing flash attention op");
  }
  return probe;
}

bool prepare_emel_session(const emel_fixture &fixture, emel_session &session) {
  session.model_data = fixture.model_data;
  session.formatter_binding = fixture.formatter_binding;
  if (emel::model::generation::build_contract(session.model_data,
                                              session.generation_contract) !=
      emel::error::cast(emel::model::loader::error::none)) {
    return false;
  }
  const auto matmul_policy =
      emel::kernel::matmul::make_auto_execution_policy(
          session.parallel_matmul_lanes);
  session.generator = std::make_unique<emel::text::generator::sm>(
      emel::text::generator::dependencies{
          .generation_contract = session.generation_contract,
          .conditioner = session.conditioner,
          .matmul_policy = matmul_policy,
          .runtime_policy =
              emel::tools::generation_route::make_current_runtime_policy(
                  session.model_data),
          .formatter_ctx = session.formatter_binding.formatter_ctx,
          .format_prompt = session.formatter_binding.format_prompt,
      });
  return true;
}

bool initialize_emel_session(emel_session &session,
                             const generation_case_spec &spec) {
  if (session.generator == nullptr) {
    return false;
  }

  std::string formatted_prompt = {};
  if (!emel::tools::generation_formatter_contract::format_single_user_prompt(
          session.formatter_binding, spec.prompt, formatted_prompt)) {
    return false;
  }

  const int32_t prompt_capacity =
      std::max<int32_t>(32, static_cast<int32_t>(formatted_prompt.size()) + 8);
  const int32_t decode_capacity = std::max<int32_t>(4, spec.max_tokens);
  // Reserve whole memory-contract blocks for the session budget, capped at the
  // model context window: the generator rejects pools larger than physical KV
  // capacity (emel::memory::view geometry contract).
  const int32_t session_tokens = std::min<int32_t>(
      prompt_capacity + decode_capacity, session.model_data.params.n_ctx);
  const int32_t block_capacity = std::max<int32_t>(
      1, emel::memory::view::blocks_for_tokens(
             emel::memory::view::DEFAULT_BLOCK_TOKENS, session_tokens));

  reset_initialize_capture(session);
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  emel::text::generator::event::initialize request{
      &session.tokenizer,
      tokenizer_bind_dispatch,
      tokenizer_tokenize_dispatch,
      std::span<emel::logits::sampler::fn>{},
  };
  request.preprocessor_variant =
      generation_preprocessor_variant(session.model_data);
  request.encoder_variant = generation_encoder_variant(session.model_data);
  request.add_special = false;
  request.parse_special = false;
  request.selection_mode =
      emel::text::generator::selection_mode::preselected_argmax;
  request.max_prompt_tokens = prompt_capacity;
  request.max_generated_tokens = decode_capacity;
  request.max_blocks = block_capacity;
  request.block_tokens = emel::memory::view::DEFAULT_BLOCK_TOKENS;
  request.strip_leading_space = false;
  request.error_out = &error_out;
  request.on_done = {&session, on_initialize_done};
  request.on_error = {&session, on_initialize_error};

  const bool accepted = session.generator->process_event(request);
  if ((!accepted || !session.initialize.done || session.initialize.error ||
       error_out != emel::error::cast(emel::text::generator::error::none)) &&
      std::getenv("EMEL_DEBUG_GENERATION_BENCH") != nullptr) {
    std::fprintf(
        stderr,
        "initialize_emel_session debug accepted=%d done=%d error=%d "
        "callback_err=%s "
        "event_err=%s arch=%.*s formatter=%.*s case=%s\n",
        accepted ? 1 : 0, session.initialize.done ? 1 : 0,
        session.initialize.error ? 1 : 0,
        generator_error_name(session.initialize.err).data(),
        generator_error_name(error_out).data(),
        static_cast<int>(
            emel::model::architecture_name_view(session.model_data).size()),
        emel::model::architecture_name_view(session.model_data).data(),
        static_cast<int>(session.formatter_binding.contract.size()),
        session.formatter_binding.contract.data(), spec.name.data());
  }
  return accepted && session.initialize.done && !session.initialize.error &&
         error_out == emel::error::cast(emel::text::generator::error::none);
}

bool run_emel_generate(emel_session &session, const generation_case_spec &spec,
                       generation_result &result_out) {
  if (session.generator == nullptr) {
    return false;
  }

  result_out = {};
  reset_generation_capture(session);
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  std::array<emel::text::formatter::chat_message, 1> message_storage = {};
  emel::text::generator::event::generate request{
      emel::tools::generation_formatter_contract::single_user_messages(
          message_storage, spec.prompt),
      spec.max_tokens,
      std::span<char>{result_out.output},
      result_out.output_length,
  };
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  request.generated_token_ids_out = std::span<int32_t>{
      result_out.output_token_ids.data(), result_out.output_token_ids.size()};
  request.error_out = &error_out;
  request.on_done = {&session, on_generation_done};
  request.on_error = {&session, on_generation_error};
  const bool accepted = session.generator->process_event(request);
  if ((!accepted || !session.generation.done || session.generation.error ||
       error_out != emel::error::cast(emel::text::generator::error::none)) &&
      std::getenv("EMEL_DEBUG_GENERATION_BENCH") != nullptr) {
    std::fprintf(
        stderr,
        "run_emel_generate debug accepted=%d done=%d error=%d callback_err=%s "
        "event_err=%s arch=%.*s formatter=%.*s case=%s\n",
        accepted ? 1 : 0, session.generation.done ? 1 : 0,
        session.generation.error ? 1 : 0,
        generator_error_name(session.generation.err).data(),
        generator_error_name(error_out).data(),
        static_cast<int>(
            emel::model::architecture_name_view(session.model_data).size()),
        emel::model::architecture_name_view(session.model_data).data(),
        static_cast<int>(session.formatter_binding.contract.size()),
        session.formatter_binding.contract.data(), spec.name.data());
  }
  if (!accepted || !session.generation.done || session.generation.error ||
      error_out != emel::error::cast(emel::text::generator::error::none)) {
    return false;
  }

  result_out.tokens_generated = session.generation.tokens_generated;
  result_out.output_token_ids_count = std::min<int32_t>(
      session.generation.tokens_generated,
      static_cast<int32_t>(result_out.output_token_ids.size()));
  result_out.output_length = session.generation.output_length;
  return true;
}

llama_token select_argmax_token_from_logits(const float *logits,
                                            const int32_t vocab_size) {
  int32_t best_index = 0;
  float best_score = logits[0];
  for (int32_t idx = 1; idx < vocab_size; ++idx) {
    if (logits[idx] > best_score) {
      best_score = logits[idx];
      best_index = idx;
    }
  }
  return static_cast<llama_token>(best_index);
}

bool tokenize_reference_prompt(const reference_fixture &fixture,
                               const generation_case_spec &spec,
                               generation_seam_audit &seam,
                               std::vector<llama_token> &tokens_out) {
  if (fixture.vocab == nullptr) {
    return false;
  }

  std::string formatted_prompt = {};
  if (!format_direct_reference_prompt(seam, fixture.formatter, spec.prompt,
                                      formatted_prompt)) {
    return false;
  }

  int32_t token_capacity =
      std::max<int32_t>(8, static_cast<int32_t>(formatted_prompt.size()) + 8);
  tokens_out.resize(static_cast<size_t>(token_capacity));
  int32_t token_count = tokenize_direct_reference_prompt(
      seam, fixture.vocab, formatted_prompt, tokens_out.data(), token_capacity);
  if (token_count < 0) {
    token_capacity = -token_count;
    tokens_out.resize(static_cast<size_t>(token_capacity));
    token_count =
        tokenize_direct_reference_prompt(seam, fixture.vocab, formatted_prompt,
                                         tokens_out.data(), token_capacity);
  }
  if (token_count <= 0) {
    return false;
  }

  tokens_out.resize(static_cast<size_t>(token_count));
  return true;
}

bool append_reference_piece(const reference_fixture &fixture,
                            generation_seam_audit &seam,
                            const llama_token token,
                            generation_result &result_out) {
  if (fixture.vocab == nullptr ||
      result_out.output_length >= result_out.output.size()) {
    return false;
  }

  if (reference_vocab_is_control(seam, fixture.vocab, token) ||
      reference_vocab_is_eog(seam, fixture.vocab, token)) {
    return true;
  }

  const char *piece = reference_vocab_text(seam, fixture.vocab, token);
  if (piece == nullptr) {
    return false;
  }

  const size_t piece_len = std::strlen(piece);
  if (result_out.output_length + piece_len > result_out.output.size()) {
    return false;
  }

  if (piece_len > 0u) {
    std::memcpy(result_out.output.data() + result_out.output_length, piece,
                piece_len);
  }
  result_out.output_length += piece_len;
  return true;
}

bool run_reference_generate(const reference_fixture &fixture,
                            const generation_case_spec &spec,
                            generation_seam_audit &seam,
                            generation_result &result_out) {
  if (fixture.model == nullptr || fixture.context == nullptr ||
      fixture.vocab == nullptr || fixture.vocab_size <= 0) {
    return false;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(fixture, spec, seam, prompt_tokens)) {
    return false;
  }

  return run_reference_generate_preloaded(fixture, spec, fixture.context.get(),
                                          prompt_tokens, seam, result_out);
}

bool reset_reference_context(llama_context *ctx) {
  if (ctx == nullptr) {
    return false;
  }

  const llama_memory_t memory = llama_get_memory(ctx);
  if (memory == nullptr) {
    return false;
  }

  llama_memory_clear(memory, false);
  return true;
}

bool run_reference_generate_preloaded(
    const reference_fixture &fixture, const generation_case_spec &spec,
    llama_context *ctx, const std::vector<llama_token> &prompt_tokens,
    generation_seam_audit &seam, generation_result &result_out) {
  if (fixture.vocab == nullptr || fixture.vocab_size <= 0 || ctx == nullptr ||
      prompt_tokens.empty()) {
    return false;
  }

  if (!reset_reference_context(ctx)) {
    return false;
  }

  result_out = {};
  llama_batch prompt_batch =
      llama_batch_get_one(const_cast<llama_token *>(prompt_tokens.data()),
                          static_cast<int32_t>(prompt_tokens.size()));
  if (run_direct_reference_decode(seam, ctx, prompt_batch) != 0) {
    return false;
  }

  for (int32_t step = 0; step < spec.max_tokens; ++step) {
    float *logits = read_direct_reference_logits(seam, ctx);
    if (logits == nullptr) {
      return false;
    }

    const llama_token selected =
        select_argmax_token_from_logits(logits, fixture.vocab_size);
    append_generation_token_id(result_out, static_cast<int32_t>(selected));
    result_out.tokens_generated += 1;
    if (!append_reference_piece(fixture, seam, selected, result_out)) {
      return false;
    }
    if (reference_vocab_is_eog(seam, fixture.vocab, selected)) {
      break;
    }

    llama_token next_token = selected;
    llama_batch decode_batch = llama_batch_get_one(&next_token, 1);
    if (run_direct_reference_decode(seam, ctx, decode_batch) != 0) {
      return false;
    }
  }
  return true;
}

using steady_clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(const steady_clock::time_point start,
                         const steady_clock::time_point end) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count());
}

std::uint64_t saturating_remainder(const std::uint64_t total,
                                   const std::uint64_t part0,
                                   const std::uint64_t part1,
                                   const std::uint64_t part2,
                                   const std::uint64_t part3) noexcept {
  const std::uint64_t measured = part0 + part1 + part2 + part3;
  return measured >= total ? 0u : total - measured;
}

enum class reference_prefill_probe_bucket : uint8_t {
  linear = 0u,
  attention = 1u,
  misc = 2u,
};

reference_prefill_probe_bucket
classify_reference_prefill_tensor(const ggml_tensor *tensor) {
  if (tensor == nullptr) {
    return reference_prefill_probe_bucket::misc;
  }

  const char *op_name = ggml_op_name(tensor->op);
  if (op_name == nullptr) {
    return reference_prefill_probe_bucket::misc;
  }
  if (std::strcmp(op_name, "MUL_MAT") == 0 ||
      std::strcmp(op_name, "MUL_MAT_ID") == 0) {
    return reference_prefill_probe_bucket::linear;
  }
  if (std::strcmp(op_name, "FLASH_ATTN_EXT") == 0) {
    return reference_prefill_probe_bucket::attention;
  }
  return reference_prefill_probe_bucket::misc;
}

struct reference_prefill_probe_state {
  prefill_probe_breakdown breakdown = {};
  steady_clock::time_point current_start = {};
  reference_prefill_probe_bucket current_bucket =
      reference_prefill_probe_bucket::misc;
  bool current_pending = false;
};

bool observe_reference_prefill_node(ggml_tensor *tensor, const bool ask,
                                    void *user_data) {
  auto *state = static_cast<reference_prefill_probe_state *>(user_data);
  if (state == nullptr) {
    return false;
  }

  if (ask) {
    state->current_start = steady_clock::now();
    state->current_bucket = classify_reference_prefill_tensor(tensor);
    state->current_pending = true;
    return true;
  }

  if (!state->current_pending) {
    return true;
  }

  const std::uint64_t elapsed =
      elapsed_ns(state->current_start, steady_clock::now());
  switch (state->current_bucket) {
  case reference_prefill_probe_bucket::linear:
    state->breakdown.linear_ns += elapsed;
    break;
  case reference_prefill_probe_bucket::attention:
    state->breakdown.attention_ns += elapsed;
    break;
  case reference_prefill_probe_bucket::misc:
    state->breakdown.misc_ns += elapsed;
    break;
  }
  state->current_pending = false;
  return true;
}

bool tokenize_conditioned_prompt(emel_session &session,
                                 const generation_case_spec &spec,
                                 std::vector<int32_t> &tokens_out) {
  const int32_t token_capacity = std::max<int32_t>(
      1024, static_cast<int32_t>(spec.prompt.size()) * 8 + 64);
  tokens_out.assign(static_cast<size_t>(token_capacity), 0);

  int32_t token_count = 0;
  int32_t conditioned_error = 0;
  std::array<emel::text::formatter::chat_message, 1> message_storage = {};
  emel::text::conditioner::event::prepare prepare_ev{token_count,
                                                     conditioned_error};
  prepare_ev.messages =
      emel::tools::generation_formatter_contract::single_user_messages(
          message_storage, spec.prompt);
  prepare_ev.add_generation_prompt = true;
  prepare_ev.enable_thinking = false;
  prepare_ev.token_ids_out = tokens_out.data();
  prepare_ev.token_capacity = token_capacity;
  const bool accepted = session.conditioner.process_event(prepare_ev);
  if (!accepted || conditioned_error != 0 || token_count <= 0 ||
      token_count > token_capacity) {
    return false;
  }

  tokens_out.resize(static_cast<size_t>(token_count));
  return true;
}

bool measure_emel_stage_probe(emel_session &session,
                              const generation_case_spec &spec,
                              emel::bench::generation_stage_probe &probe_out) {
  auto total_start = steady_clock::now();
  generation_result total_result = {};
  if (!run_emel_generate(session, spec, total_result)) {
    return false;
  }
  probe_out.emel_total_ns = elapsed_ns(total_start, steady_clock::now());

  std::vector<int32_t> prompt_tokens;
  const auto conditioning_start = steady_clock::now();
  if (!tokenize_conditioned_prompt(session, spec, prompt_tokens) ||
      prompt_tokens.empty()) {
    return false;
  }
  probe_out.emel_conditioning_ns =
      elapsed_ns(conditioning_start, steady_clock::now());
  probe_out.emel_prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
  probe_out.emel_prefill_contract = "actor_public_generate";
  probe_out.emel_prefill_step_size = 0;
  probe_out.emel_unattributed_ns = saturating_remainder(
      probe_out.emel_total_ns, probe_out.emel_conditioning_ns,
      probe_out.emel_prefill_ns, probe_out.emel_first_decode_ns,
      probe_out.emel_steady_decode_ns);
  return true;
}

bool measure_reference_prefill_probe(
    const reference_fixture &fixture,
    const std::vector<llama_token> &prompt_tokens,
    prefill_probe_breakdown &breakdown_out) {
  if (fixture.model == nullptr || prompt_tokens.empty()) {
    return false;
  }

  reference_prefill_probe_state probe_state = {};
  llama_context_ptr probe_ctx = make_reference_context(
      fixture.model.get(), observe_reference_prefill_node, &probe_state);
  if (probe_ctx == nullptr || !reset_reference_context(probe_ctx.get())) {
    return false;
  }

  generation_seam_audit probe_seam = {};
  llama_batch prompt_batch =
      llama_batch_get_one(const_cast<llama_token *>(prompt_tokens.data()),
                          static_cast<int32_t>(prompt_tokens.size()));
  if (run_direct_reference_decode(probe_seam, probe_ctx.get(), prompt_batch) !=
      0) {
    return false;
  }

  breakdown_out = probe_state.breakdown;
  return true;
}

bool measure_reference_stage_probe(
    const reference_fixture &fixture, const generation_case_spec &spec,
    emel::bench::generation_stage_probe &probe_out) {
  generation_seam_audit total_seam = {};
  auto total_start = steady_clock::now();
  generation_result total_result = {};
  if (!run_reference_generate(fixture, spec, total_seam, total_result)) {
    return false;
  }
  probe_out.reference_total_ns = elapsed_ns(total_start, steady_clock::now());

  generation_seam_audit seam = {};
  std::vector<llama_token> prompt_tokens;
  const auto conditioning_start = steady_clock::now();
  if (!tokenize_reference_prompt(fixture, spec, seam, prompt_tokens) ||
      prompt_tokens.empty()) {
    return false;
  }
  probe_out.reference_conditioning_ns =
      elapsed_ns(conditioning_start, steady_clock::now());

  if (fixture.context == nullptr) {
    return false;
  }
  llama_context *ctx = fixture.context.get();
  if (!reset_reference_context(ctx)) {
    return false;
  }

  generation_result result_out = {};
  const auto prefill_start = steady_clock::now();
  llama_batch prompt_batch = llama_batch_get_one(
      prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
  if (run_direct_reference_decode(seam, ctx, prompt_batch) != 0) {
    return false;
  }
  probe_out.reference_prefill_ns =
      elapsed_ns(prefill_start, steady_clock::now());
  if (spec.max_tokens == 1) {
    prefill_probe_breakdown prefill_breakdown = {};
    if (!measure_reference_prefill_probe(fixture, prompt_tokens,
                                         prefill_breakdown)) {
      return false;
    }
    probe_out.reference_prefill_linear_probe_ns = prefill_breakdown.linear_ns;
    probe_out.reference_prefill_attention_probe_ns =
        prefill_breakdown.attention_ns;
    probe_out.reference_prefill_misc_probe_ns = prefill_breakdown.misc_ns;
  }

  for (int32_t step = 0; step < spec.max_tokens; ++step) {
    float *logits = read_direct_reference_logits(seam, ctx);
    if (logits == nullptr) {
      return false;
    }

    const llama_token selected =
        select_argmax_token_from_logits(logits, fixture.vocab_size);
    append_generation_token_id(result_out, static_cast<int32_t>(selected));
    result_out.tokens_generated += 1;
    if (!append_reference_piece(fixture, seam, selected, result_out)) {
      return false;
    }
    if (reference_vocab_is_eog(seam, fixture.vocab, selected)) {
      break;
    }

    const auto decode_start = steady_clock::now();
    llama_token next_token = selected;
    llama_batch decode_batch = llama_batch_get_one(&next_token, 1);
    if (run_direct_reference_decode(seam, ctx, decode_batch) != 0) {
      return false;
    }
    const auto decode_ns = elapsed_ns(decode_start, steady_clock::now());
    if (step == 0) {
      probe_out.reference_first_decode_ns = decode_ns;
    } else {
      probe_out.reference_steady_decode_ns += decode_ns;
    }
  }

  probe_out.reference_unattributed_ns = saturating_remainder(
      probe_out.reference_total_ns, probe_out.reference_conditioning_ns,
      probe_out.reference_prefill_ns, probe_out.reference_first_decode_ns,
      probe_out.reference_steady_decode_ns);
  return true;
}

bool capture_generation_stage_probe(emel_session &session,
                                    const reference_fixture &fixture,
                                    const generation_case_spec &spec) {
  emel::bench::generation_stage_probe probe = {};
  probe.name = std::string(spec.name);
  probe.benchmark_lane = std::string{generation_benchmark_lane_name()};
  return measure_emel_stage_probe(session, spec, probe) &&
         measure_reference_stage_probe(fixture, spec, probe) &&
         (g_generation_stage_probes.push_back(std::move(probe)), true);
}

emel::bench::config generation_case_config(const emel::bench::config &cfg) {
  emel::bench::config case_cfg = cfg;
  case_cfg.iterations = read_env_u64("EMEL_BENCH_GENERATION_ITERS", 1u);
  case_cfg.runs = read_env_size("EMEL_BENCH_GENERATION_RUNS", cfg.runs);
  case_cfg.warmup_iterations =
      read_env_u64("EMEL_BENCH_GENERATION_WARMUP_ITERS", 0u);
  case_cfg.warmup_runs = read_env_size("EMEL_BENCH_GENERATION_WARMUP_RUNS", 0u);
  return case_cfg;
}

void prepare_emel_generation_fixture(
    const generation_fixture_spec &spec,
    prepared_emel_generation_fixture &prepared_fixture) {
  prepared_fixture.spec = &spec;
  prepared_fixture.cases = generation_cases_for_fixture(*spec.fixture, false);
  g_generation_fixture_rel = spec.fixture->fixture_rel;

  if (prepared_fixture.cases.empty()) {
    fail_bench_setup("prepare_emel_generation_fixture",
                     "no workload manifests for fixture");
  }
  for (const generation_case_spec &generation_case : prepared_fixture.cases) {
    validate_generation_workload_fixture(*spec.fixture, generation_case);
  }

  const std::string model_path =
      resolve_generation_model_path(spec.fixture->fixture_rel);
  if (!prepare_emel_fixture(prepared_fixture.emel, model_path)) {
    if (prepared_fixture.emel.load.error) {
      fail_bench_setup(
          "prepare_emel_fixture",
          model_loader_error_name(prepared_fixture.emel.load.err).data());
    }
    if (!prepared_fixture.emel.formatter_binding.contract.empty()) {
      g_generation_formatter_contract.assign(
          prepared_fixture.emel.formatter_binding.contract);
      fail_bench_setup("prepare_emel_fixture",
                       prepared_fixture.emel.formatter_binding.contract.data());
    }
    fail_bench_setup("prepare_emel_fixture", model_path.c_str());
  }

  if (!emel::model::detail::load_vocab_from_gguf(
          kv_binding_from_fixture(prepared_fixture.emel),
          prepared_fixture.emel.model_data.vocab_data)) {
    fail_bench_setup("load_vocab_from_gguf", model_path.c_str());
  }
  prepared_fixture.emel.model_data.params.n_vocab = static_cast<int32_t>(
      prepared_fixture.emel.model_data.vocab_data.n_tokens);
}

void prepare_reference_generation_fixture(
    const generation_fixture_spec &spec,
    prepared_reference_generation_fixture &prepared_fixture) {
  ensure_llama_backend_ready();

  prepared_fixture.spec = &spec;
  prepared_fixture.cases = generation_cases_for_fixture(*spec.fixture, true);
  g_generation_fixture_rel = spec.fixture->fixture_rel;

  if (prepared_fixture.cases.empty()) {
    fail_bench_setup("prepare_reference_generation_fixture",
                     "no comparable workload manifests");
  }
  for (const generation_case_spec &generation_case : prepared_fixture.cases) {
    validate_generation_workload_fixture(*spec.fixture, generation_case);
  }

  const std::string model_path =
      resolve_generation_model_path(spec.fixture->fixture_rel);
  if (!prepare_reference_fixture(prepared_fixture.reference, model_path)) {
    fail_bench_setup("prepare_reference_fixture", model_path.c_str());
  }
}

void prepare_compare_generation_fixture(
    const generation_fixture_spec &spec,
    prepared_generation_fixture &prepared_fixture) {
  prepared_fixture.spec = &spec;
  prepared_fixture.cases = generation_cases_for_fixture(*spec.fixture, true);
  g_generation_fixture_rel = spec.fixture->fixture_rel;

  if (prepared_fixture.cases.empty()) {
    fail_bench_setup("prepare_compare_generation_fixture",
                     "no comparable workload manifests");
  }
  for (const generation_case_spec &generation_case : prepared_fixture.cases) {
    validate_generation_workload_fixture(*spec.fixture, generation_case);
  }

  const std::string model_path =
      resolve_generation_model_path(spec.fixture->fixture_rel);
  if (!prepare_emel_fixture(prepared_fixture.emel, model_path)) {
    if (prepared_fixture.emel.load.error) {
      fail_bench_setup(
          "prepare_emel_fixture",
          model_loader_error_name(prepared_fixture.emel.load.err).data());
    }
    if (!prepared_fixture.emel.formatter_binding.contract.empty()) {
      g_generation_formatter_contract.assign(
          prepared_fixture.emel.formatter_binding.contract);
      fail_bench_setup("prepare_emel_fixture",
                       prepared_fixture.emel.formatter_binding.contract.data());
    }
    fail_bench_setup("prepare_emel_fixture", model_path.c_str());
  }
  if (!emel::model::detail::load_vocab_from_gguf(
          kv_binding_from_fixture(prepared_fixture.emel),
          prepared_fixture.emel.model_data.vocab_data)) {
    fail_bench_setup("load_vocab_from_gguf", model_path.c_str());
  }
  prepared_fixture.emel.model_data.params.n_vocab = static_cast<int32_t>(
      prepared_fixture.emel.model_data.vocab_data.n_tokens);

  ensure_llama_backend_ready();
  if (!prepare_reference_fixture(prepared_fixture.reference, model_path)) {
    fail_bench_setup("prepare_reference_fixture", model_path.c_str());
  }
}

const std::vector<prepared_emel_generation_fixture> &
maintained_emel_generation_fixtures() {
  static const std::vector<prepared_emel_generation_fixture> fixtures = [] {
    std::vector<prepared_emel_generation_fixture> prepared_fixtures = {};
    prepared_fixtures.reserve(k_emel_generation_fixtures.size());
    for (size_t fixture_index = 0u;
         fixture_index < k_emel_generation_fixtures.size(); ++fixture_index) {
      const generation_fixture_spec &spec =
          k_emel_generation_fixtures[fixture_index];
      if (!generation_fixture_exists(*spec.fixture)) {
        report_missing_generation_fixture(*spec.fixture);
        continue;
      }
      prepared_fixtures.emplace_back();
      prepare_emel_generation_fixture(spec, prepared_fixtures.back());
    }
    return prepared_fixtures;
  }();
  return fixtures;
}

const std::vector<prepared_reference_generation_fixture> &
maintained_reference_generation_fixtures() {
  static const std::vector<prepared_reference_generation_fixture> fixtures =
      [] {
        std::vector<prepared_reference_generation_fixture> prepared_fixtures =
            {};
        prepared_fixtures.reserve(k_compare_generation_fixtures.size());
        for (size_t fixture_index = 0u;
             fixture_index < k_compare_generation_fixtures.size();
             ++fixture_index) {
          const generation_fixture_spec &spec =
              k_compare_generation_fixtures[fixture_index];
          if (!generation_fixture_exists(*spec.fixture)) {
            report_missing_generation_fixture(*spec.fixture);
            continue;
          }
          prepared_fixtures.emplace_back();
          prepare_reference_generation_fixture(spec, prepared_fixtures.back());
        }
        return prepared_fixtures;
      }();
  return fixtures;
}

const std::vector<prepared_generation_fixture> &
maintained_compare_generation_fixtures() {
  static const std::vector<prepared_generation_fixture> fixtures = [] {
    std::vector<prepared_generation_fixture> prepared_fixtures = {};
    prepared_fixtures.reserve(k_compare_generation_fixtures.size());
    for (size_t fixture_index = 0u;
         fixture_index < k_compare_generation_fixtures.size();
         ++fixture_index) {
      const generation_fixture_spec &spec =
          k_compare_generation_fixtures[fixture_index];
      if (!generation_fixture_exists(*spec.fixture)) {
        report_missing_generation_fixture(*spec.fixture);
        continue;
      }
      prepared_fixtures.emplace_back();
      prepare_compare_generation_fixture(spec, prepared_fixtures.back());
    }
    return prepared_fixtures;
  }();
  return fixtures;
}

} // namespace

namespace emel::bench {

void set_generation_lane_mode(const generation_lane_mode mode) noexcept {
  g_generation_lane_mode = mode;
}

generation_lane_mode generation_lane_mode_current() noexcept {
  return g_generation_lane_mode;
}

std::string_view generation_formatter_contract() noexcept {
  return g_generation_formatter_contract;
}

bool generation_flash_evidence_ready() noexcept {
  return g_generation_flash_evidence.ready;
}

std::uint64_t generation_flash_evidence_dispatch_calls() noexcept {
  return g_generation_flash_evidence.flash_dispatch_calls;
}

std::uint64_t generation_flash_evidence_optimized_dispatch_calls() noexcept {
  return g_generation_flash_evidence.optimized_flash_dispatch_calls;
}

std::uint64_t generation_flash_evidence_shared_dispatch_calls() noexcept {
  return g_generation_flash_evidence.shared_flash_dispatch_calls;
}

std::uint32_t
generation_runtime_contract_native_quantized_stage_count() noexcept {
  return g_generation_flash_evidence.native_quantized_stage_count;
}

std::uint32_t
generation_runtime_contract_approved_dense_f32_stage_count() noexcept {
  return g_generation_flash_evidence.approved_dense_f32_stage_count;
}

std::uint32_t
generation_runtime_contract_disallowed_fallback_stage_count() noexcept {
  return g_generation_flash_evidence.disallowed_fallback_stage_count;
}

std::uint32_t
generation_runtime_contract_explicit_no_claim_stage_count() noexcept {
  return g_generation_flash_evidence.explicit_no_claim_stage_count;
}

std::uint64_t
generation_quantized_evidence_native_q8_0_dispatch_calls() noexcept {
  return g_generation_flash_evidence.native_q8_0_dispatch_calls;
}

std::uint64_t
generation_quantized_evidence_packed_q8_0_dispatch_calls() noexcept {
  return g_generation_flash_evidence.packed_q8_0_dispatch_calls;
}

std::uint64_t
generation_quantized_evidence_optimized_q2_dispatch_calls() noexcept {
  return g_generation_flash_evidence.optimized_q2_dispatch_calls;
}

std::uint64_t
generation_quantized_evidence_shared_q2_dispatch_calls() noexcept {
  return g_generation_flash_evidence.shared_q2_dispatch_calls;
}

std::uint64_t
generation_quantized_evidence_optimized_q3_dispatch_calls() noexcept {
  return g_generation_flash_evidence.optimized_q3_dispatch_calls;
}

std::uint64_t
generation_quantized_evidence_shared_q3_dispatch_calls() noexcept {
  return g_generation_flash_evidence.shared_q3_dispatch_calls;
}

std::uint64_t
generation_quantized_evidence_optimized_q4_dispatch_calls() noexcept {
  return g_generation_flash_evidence.optimized_q4_dispatch_calls;
}

std::uint64_t
generation_quantized_evidence_shared_q4_dispatch_calls() noexcept {
  return g_generation_flash_evidence.shared_q4_dispatch_calls;
}

std::uint64_t
generation_quantized_evidence_optimized_q6_dispatch_calls() noexcept {
  return g_generation_flash_evidence.optimized_q6_dispatch_calls;
}

std::uint64_t
generation_quantized_evidence_shared_q6_dispatch_calls() noexcept {
  return g_generation_flash_evidence.shared_q6_dispatch_calls;
}

std::string_view generation_architecture_contract() noexcept {
  return g_generation_architecture_contract;
}

std::int32_t generation_flash_evidence_emel_decode_calls() noexcept {
  return g_generation_flash_evidence.seam.emel_reference_decode_calls;
}

std::int32_t generation_flash_evidence_emel_logits_calls() noexcept {
  return g_generation_flash_evidence.seam.emel_reference_logits_calls;
}

std::int32_t generation_flash_evidence_reference_decode_calls() noexcept {
  return g_generation_flash_evidence.seam.direct_reference_decode_calls;
}

std::int32_t generation_flash_evidence_reference_logits_calls() noexcept {
  return g_generation_flash_evidence.seam.direct_reference_logits_calls;
}

std::size_t generation_stage_probe_count() noexcept {
  return g_generation_stage_probes.size();
}

generation_stage_probe
generation_stage_probe_at(const std::size_t index) noexcept {
  return index < g_generation_stage_probes.size()
             ? g_generation_stage_probes[index]
             : generation_stage_probe{};
}

void append_emel_generation_cases_for_current_benchmark_lane(
    std::vector<result> &results, const config &cfg) {
  const bool compare_lane =
      generation_lane_mode_current() == generation_lane_mode::compare;
  const config case_cfg = generation_case_config(cfg);

  if (compare_lane) {
    const auto &fixtures = maintained_compare_generation_fixtures();
    for (size_t fixture_index = 0u; fixture_index < fixtures.size();
         ++fixture_index) {
      const prepared_generation_fixture &prepared_fixture =
          fixtures[fixture_index];
      const generation_fixture_spec *spec = prepared_fixture.spec;
      const emel_fixture &fixture = prepared_fixture.emel;
      g_generation_fixture_rel = spec->fixture->fixture_rel;
      for (const generation_case_spec &generation_case :
           prepared_fixture.cases) {
        if (!generation_workload_selected(generation_case)) {
          continue;
        }
        volatile std::size_t sink = 0u;
        generation_seam_audit seam = {};
        std::uint64_t kernel_dispatch_calls = 0u;
        std::uint64_t flash_dispatch_calls = 0u;
        std::uint64_t optimized_flash_dispatch_calls = 0u;
        std::uint64_t shared_flash_dispatch_calls = 0u;
        std::uint32_t native_quantized_stage_count = 0u;
        std::uint32_t approved_dense_f32_stage_count = 0u;
        std::uint32_t disallowed_fallback_stage_count = 0u;
        std::uint32_t explicit_no_claim_stage_count = 0u;
        std::uint64_t native_q8_0_dispatch_calls = 0u;
        std::uint64_t packed_q8_0_dispatch_calls = 0u;
        std::uint64_t optimized_q2_dispatch_calls = 0u;
        std::uint64_t shared_q2_dispatch_calls = 0u;
        std::uint64_t optimized_q3_dispatch_calls = 0u;
        std::uint64_t shared_q3_dispatch_calls = 0u;
        std::uint64_t optimized_q4_dispatch_calls = 0u;
        std::uint64_t shared_q4_dispatch_calls = 0u;
        std::uint64_t optimized_q6_dispatch_calls = 0u;
        std::uint64_t shared_q6_dispatch_calls = 0u;
        generation_result latest_generated = {};
        auto session = std::make_unique<emel_session>();
        if (!prepare_emel_session(fixture, *session)) {
          fail_bench_setup("build_generation_contract",
                           generation_case.name.data());
        }
        if (!initialize_emel_session(*session, generation_case)) {
          fail_bench_setup("initialize_emel_session",
                           generation_case.name.data());
        }
        if (!configure_generator_benchmark_lane(*session)) {
          fail_bench_setup("configure_generator_benchmark_lane",
                           generation_case.name.data());
        }
        validate_generation_formatter_contract(
            generation_case, session->formatter_binding.contract);

        auto fn = [&]() {
          reset_generation_seam(session->seam);
          emel::text::generator::diagnostics before = {};
          if (!capture_generator_diagnostics(*session, before)) {
            fail_bench_setup("capture_generator_diagnostics_before",
                             generation_case.name.data());
          }
          native_quantized_stage_count = before.native_quantized_stage_count;
          approved_dense_f32_stage_count =
              before.approved_dense_f32_stage_count;
          disallowed_fallback_stage_count =
              before.disallowed_fallback_stage_count;
          explicit_no_claim_stage_count = before.explicit_no_claim_stage_count;

          generation_result generated{};
          if (!run_emel_generate(*session, generation_case, generated)) {
            fail_bench_setup("run_emel_generate", generation_case.name.data());
          }
          latest_generated = generated;
          emel::text::generator::diagnostics after = {};
          if (!capture_generator_diagnostics(*session, after)) {
            fail_bench_setup("capture_generator_diagnostics_after",
                             generation_case.name.data());
          }
          seam = session->seam;
          kernel_dispatch_calls = after.kernel_dispatch_calls -
                                  before.kernel_dispatch_calls;
          flash_dispatch_calls = after.flash_attention_dispatch_calls -
                                 before.flash_attention_dispatch_calls;
          optimized_flash_dispatch_calls =
              after.optimized_flash_dispatch_calls -
              before.optimized_flash_dispatch_calls;
          shared_flash_dispatch_calls = after.shared_flash_dispatch_calls -
                                        before.shared_flash_dispatch_calls;
          native_q8_0_dispatch_calls = after.native_q8_0_dispatch_calls -
                                       before.native_q8_0_dispatch_calls;
          packed_q8_0_dispatch_calls = after.packed_q8_0_dispatch_calls -
                                       before.packed_q8_0_dispatch_calls;
          optimized_q2_dispatch_calls = after.optimized_q2_dispatch_calls -
                                        before.optimized_q2_dispatch_calls;
          shared_q2_dispatch_calls =
              after.shared_q2_dispatch_calls - before.shared_q2_dispatch_calls;
          optimized_q3_dispatch_calls = after.optimized_q3_dispatch_calls -
                                        before.optimized_q3_dispatch_calls;
          shared_q3_dispatch_calls =
              after.shared_q3_dispatch_calls - before.shared_q3_dispatch_calls;
          optimized_q4_dispatch_calls = after.optimized_q4_dispatch_calls -
                                        before.optimized_q4_dispatch_calls;
          shared_q4_dispatch_calls =
              after.shared_q4_dispatch_calls - before.shared_q4_dispatch_calls;
          optimized_q6_dispatch_calls = after.optimized_q6_dispatch_calls -
                                        before.optimized_q6_dispatch_calls;
          shared_q6_dispatch_calls =
              after.shared_q6_dispatch_calls - before.shared_q6_dispatch_calls;
          sink ^= generated.output_length;
        };

        const std::string case_name =
            generation_benchmark_case_name(generation_case.name);
        results.push_back(measure_case(case_name.c_str(), case_cfg, fn));
        result &compare_record = results.back();
        compare_record.compare_group = generation_case.manifest.compare_group;
        compare_record.benchmark_lane =
            std::string{generation_benchmark_lane_name()};
        compare_record.lane = "emel";
        compare_record.backend_id = "emel.generator";
        compare_record.backend_language = "cpp";
        compare_record.thread_count = generation_emel_thread_count();
        compare_record.thread_contract =
            std::string{generation_emel_thread_contract()};
        compare_record.workload_id = generation_case.manifest.id;
        compare_record.workload_manifest_path =
            generation_case.manifest.workload_manifest_path;
        compare_record.comparison_mode =
            generation_case.manifest.comparison_mode;
        compare_record.model_id = generation_case.manifest.fixture_name;
        compare_record.fixture_id = generation_case.manifest.fixture_rel;
        compare_record.prompt_fixture_id =
            generation_case.manifest.prompt_fixture_id;
        compare_record.prompt_fixture_path =
            generation_case.manifest.prompt_fixture_path;
        compare_record.prompt_id = generation_case.manifest.prompt_id;
        compare_record.formatter_mode = generation_case.manifest.formatter_mode;
        compare_record.formatter_contract =
            generation_case.manifest.formatter_contract;
        compare_record.sampling_id = generation_case.manifest.sampling_id;
        compare_record.stop_id = generation_case.manifest.stop_id;
        compare_record.seed = generation_case.manifest.seed;
        compare_record.max_output_tokens =
            generation_case.manifest.max_output_tokens;
        compare_record.comparable = generation_case.manifest.comparable;
        capture_generation_output_metrics(compare_record, latest_generated);
        compare_record.kernel_dispatch_calls = kernel_dispatch_calls;
        compare_record.flash_attention_dispatch_calls = flash_dispatch_calls;
        compare_record.optimized_flash_dispatch_calls =
            optimized_flash_dispatch_calls;
        compare_record.shared_flash_dispatch_calls =
            shared_flash_dispatch_calls;
        compare_record.native_q8_0_dispatch_calls = native_q8_0_dispatch_calls;
        compare_record.packed_q8_0_dispatch_calls =
            packed_q8_0_dispatch_calls;
        compare_record.optimized_q4_dispatch_calls =
            optimized_q4_dispatch_calls;
        compare_record.shared_q4_dispatch_calls = shared_q4_dispatch_calls;
        compare_record.optimized_q6_dispatch_calls =
            optimized_q6_dispatch_calls;
        compare_record.shared_q6_dispatch_calls = shared_q6_dispatch_calls;
        compare_record.native_quantized_stage_count =
            native_quantized_stage_count;
        compare_record.approved_dense_f32_stage_count =
            approved_dense_f32_stage_count;
        compare_record.disallowed_fallback_stage_count =
            disallowed_fallback_stage_count;
        compare_record.explicit_no_claim_stage_count =
            explicit_no_claim_stage_count;
        compare_record.note = emel::tools::append_model_load_io_strategy_note(
            generation_case.manifest.comparability_note,
            prepared_fixture.emel.load.used_io_strategy);
        if (spec->fixture->current_publication &&
            generation_case.name == k_generation_case_name) {
          g_generation_architecture_contract.assign(
              emel::model::architecture_name_view(session->model_data));
          g_generation_formatter_contract.assign(
              session->formatter_binding.contract);
          g_generation_flash_evidence.ready = true;
          g_generation_flash_evidence.flash_dispatch_calls =
              flash_dispatch_calls;
          g_generation_flash_evidence.optimized_flash_dispatch_calls =
              optimized_flash_dispatch_calls;
          g_generation_flash_evidence.shared_flash_dispatch_calls =
              shared_flash_dispatch_calls;
          g_generation_flash_evidence.native_quantized_stage_count =
              native_quantized_stage_count;
          g_generation_flash_evidence.approved_dense_f32_stage_count =
              approved_dense_f32_stage_count;
          g_generation_flash_evidence.disallowed_fallback_stage_count =
              disallowed_fallback_stage_count;
          g_generation_flash_evidence.explicit_no_claim_stage_count =
              explicit_no_claim_stage_count;
          g_generation_flash_evidence.native_q8_0_dispatch_calls =
              native_q8_0_dispatch_calls;
          g_generation_flash_evidence.packed_q8_0_dispatch_calls =
              packed_q8_0_dispatch_calls;
          g_generation_flash_evidence.optimized_q2_dispatch_calls =
              optimized_q2_dispatch_calls;
          g_generation_flash_evidence.shared_q2_dispatch_calls =
              shared_q2_dispatch_calls;
          g_generation_flash_evidence.optimized_q3_dispatch_calls =
              optimized_q3_dispatch_calls;
          g_generation_flash_evidence.shared_q3_dispatch_calls =
              shared_q3_dispatch_calls;
          g_generation_flash_evidence.optimized_q4_dispatch_calls =
              optimized_q4_dispatch_calls;
          g_generation_flash_evidence.shared_q4_dispatch_calls =
              shared_q4_dispatch_calls;
          g_generation_flash_evidence.optimized_q6_dispatch_calls =
              optimized_q6_dispatch_calls;
          g_generation_flash_evidence.shared_q6_dispatch_calls =
              shared_q6_dispatch_calls;
          g_generation_flash_evidence.seam = seam;
        }
        if (generation_seam_audit_enabled()) {
          print_generation_seam_audit("emel", seam);
          verify_emel_generation_seam(seam);
        }
        if (should_capture_generation_stage_probe(*spec, generation_case) &&
            !capture_generation_stage_probe(
                *session, prepared_fixture.reference, generation_case)) {
          fail_bench_setup("capture_generation_stage_probe",
                           generation_case.name.data());
        }
        static_cast<void>(sink);
      }
    }
    return;
  }

  const auto &fixtures = maintained_emel_generation_fixtures();
  for (const prepared_emel_generation_fixture &prepared_fixture : fixtures) {
    const generation_fixture_spec *spec = prepared_fixture.spec;
    const emel_fixture &fixture = prepared_fixture.emel;
    g_generation_fixture_rel = spec->fixture->fixture_rel;
    for (const generation_case_spec &generation_case : prepared_fixture.cases) {
      if (!generation_workload_selected(generation_case)) {
        continue;
      }
      volatile std::size_t sink = 0u;
      generation_seam_audit seam = {};
      std::uint64_t kernel_dispatch_calls = 0u;
      std::uint64_t flash_dispatch_calls = 0u;
      std::uint64_t optimized_flash_dispatch_calls = 0u;
      std::uint64_t shared_flash_dispatch_calls = 0u;
      std::uint32_t native_quantized_stage_count = 0u;
      std::uint32_t approved_dense_f32_stage_count = 0u;
      std::uint32_t disallowed_fallback_stage_count = 0u;
      std::uint32_t explicit_no_claim_stage_count = 0u;
      std::uint64_t native_q8_0_dispatch_calls = 0u;
      std::uint64_t packed_q8_0_dispatch_calls = 0u;
      std::uint64_t optimized_q2_dispatch_calls = 0u;
      std::uint64_t shared_q2_dispatch_calls = 0u;
      std::uint64_t optimized_q3_dispatch_calls = 0u;
      std::uint64_t shared_q3_dispatch_calls = 0u;
      std::uint64_t optimized_q4_dispatch_calls = 0u;
      std::uint64_t shared_q4_dispatch_calls = 0u;
      std::uint64_t optimized_q6_dispatch_calls = 0u;
      std::uint64_t shared_q6_dispatch_calls = 0u;
      generation_result latest_generated = {};
      auto session = std::make_unique<emel_session>();
      if (!prepare_emel_session(fixture, *session)) {
        fail_bench_setup("build_generation_contract",
                         generation_case.name.data());
      }
      if (!initialize_emel_session(*session, generation_case)) {
        fail_bench_setup("initialize_emel_session",
                         generation_case.name.data());
      }
      if (!configure_generator_benchmark_lane(*session)) {
        fail_bench_setup("configure_generator_benchmark_lane",
                         generation_case.name.data());
      }
      validate_generation_formatter_contract(
          generation_case, session->formatter_binding.contract);

      auto fn = [&]() {
        reset_generation_seam(session->seam);
        emel::text::generator::diagnostics before = {};
        if (!capture_generator_diagnostics(*session, before)) {
          fail_bench_setup("capture_generator_diagnostics_before",
                           generation_case.name.data());
        }
        native_quantized_stage_count = before.native_quantized_stage_count;
        approved_dense_f32_stage_count = before.approved_dense_f32_stage_count;
        disallowed_fallback_stage_count =
            before.disallowed_fallback_stage_count;
        explicit_no_claim_stage_count = before.explicit_no_claim_stage_count;

        generation_result generated{};
        if (!run_emel_generate(*session, generation_case, generated)) {
          fail_bench_setup("run_emel_generate", generation_case.name.data());
        }
        latest_generated = generated;
        emel::text::generator::diagnostics after = {};
        if (!capture_generator_diagnostics(*session, after)) {
          fail_bench_setup("capture_generator_diagnostics_after",
                           generation_case.name.data());
        }
        seam = session->seam;
        kernel_dispatch_calls = after.kernel_dispatch_calls -
                                before.kernel_dispatch_calls;
        flash_dispatch_calls = after.flash_attention_dispatch_calls -
                               before.flash_attention_dispatch_calls;
        optimized_flash_dispatch_calls = after.optimized_flash_dispatch_calls -
                                         before.optimized_flash_dispatch_calls;
        shared_flash_dispatch_calls = after.shared_flash_dispatch_calls -
                                      before.shared_flash_dispatch_calls;
        native_q8_0_dispatch_calls = after.native_q8_0_dispatch_calls -
                                     before.native_q8_0_dispatch_calls;
        packed_q8_0_dispatch_calls = after.packed_q8_0_dispatch_calls -
                                     before.packed_q8_0_dispatch_calls;
        optimized_q2_dispatch_calls = after.optimized_q2_dispatch_calls -
                                      before.optimized_q2_dispatch_calls;
        shared_q2_dispatch_calls =
            after.shared_q2_dispatch_calls - before.shared_q2_dispatch_calls;
        optimized_q3_dispatch_calls = after.optimized_q3_dispatch_calls -
                                      before.optimized_q3_dispatch_calls;
        shared_q3_dispatch_calls =
            after.shared_q3_dispatch_calls - before.shared_q3_dispatch_calls;
        optimized_q4_dispatch_calls = after.optimized_q4_dispatch_calls -
                                      before.optimized_q4_dispatch_calls;
        shared_q4_dispatch_calls =
            after.shared_q4_dispatch_calls - before.shared_q4_dispatch_calls;
        optimized_q6_dispatch_calls = after.optimized_q6_dispatch_calls -
                                      before.optimized_q6_dispatch_calls;
        shared_q6_dispatch_calls =
            after.shared_q6_dispatch_calls - before.shared_q6_dispatch_calls;
        sink ^= generated.output_length;
      };

      const std::string case_name =
          generation_benchmark_case_name(generation_case.name);
      results.push_back(measure_case(case_name.c_str(), case_cfg, fn));
      result &compare_record = results.back();
      compare_record.compare_group = generation_case.manifest.compare_group;
      compare_record.benchmark_lane =
          std::string{generation_benchmark_lane_name()};
      compare_record.lane = "emel";
      compare_record.backend_id = "emel.generator";
      compare_record.backend_language = "cpp";
      compare_record.thread_count = generation_emel_thread_count();
      compare_record.thread_contract =
          std::string{generation_emel_thread_contract()};
      compare_record.workload_id = generation_case.manifest.id;
      compare_record.workload_manifest_path =
          generation_case.manifest.workload_manifest_path;
      compare_record.comparison_mode = generation_case.manifest.comparison_mode;
      compare_record.model_id = generation_case.manifest.fixture_name;
      compare_record.fixture_id = generation_case.manifest.fixture_rel;
      compare_record.prompt_fixture_id =
          generation_case.manifest.prompt_fixture_id;
      compare_record.prompt_fixture_path =
          generation_case.manifest.prompt_fixture_path;
      compare_record.prompt_id = generation_case.manifest.prompt_id;
      compare_record.formatter_mode = generation_case.manifest.formatter_mode;
      compare_record.formatter_contract =
          generation_case.manifest.formatter_contract;
      compare_record.sampling_id = generation_case.manifest.sampling_id;
      compare_record.stop_id = generation_case.manifest.stop_id;
      compare_record.seed = generation_case.manifest.seed;
      compare_record.max_output_tokens =
          generation_case.manifest.max_output_tokens;
      compare_record.comparable = generation_case.manifest.comparable;
      capture_generation_output_metrics(compare_record, latest_generated);
      compare_record.kernel_dispatch_calls = kernel_dispatch_calls;
      compare_record.flash_attention_dispatch_calls = flash_dispatch_calls;
      compare_record.optimized_flash_dispatch_calls =
          optimized_flash_dispatch_calls;
      compare_record.shared_flash_dispatch_calls = shared_flash_dispatch_calls;
      compare_record.native_q8_0_dispatch_calls = native_q8_0_dispatch_calls;
      compare_record.packed_q8_0_dispatch_calls = packed_q8_0_dispatch_calls;
      compare_record.optimized_q4_dispatch_calls = optimized_q4_dispatch_calls;
      compare_record.shared_q4_dispatch_calls = shared_q4_dispatch_calls;
      compare_record.optimized_q6_dispatch_calls = optimized_q6_dispatch_calls;
      compare_record.shared_q6_dispatch_calls = shared_q6_dispatch_calls;
      compare_record.native_quantized_stage_count =
          native_quantized_stage_count;
      compare_record.approved_dense_f32_stage_count =
          approved_dense_f32_stage_count;
      compare_record.disallowed_fallback_stage_count =
          disallowed_fallback_stage_count;
      compare_record.explicit_no_claim_stage_count =
          explicit_no_claim_stage_count;
      compare_record.note = emel::tools::append_model_load_io_strategy_note(
          generation_case.manifest.comparability_note,
          prepared_fixture.emel.load.used_io_strategy);
      if (spec->fixture->current_publication &&
          generation_case.name == k_generation_case_name) {
        g_generation_architecture_contract.assign(
            emel::model::architecture_name_view(session->model_data));
        g_generation_formatter_contract.assign(
            session->formatter_binding.contract);
        g_generation_flash_evidence.ready = true;
        g_generation_flash_evidence.flash_dispatch_calls = flash_dispatch_calls;
        g_generation_flash_evidence.optimized_flash_dispatch_calls =
            optimized_flash_dispatch_calls;
        g_generation_flash_evidence.shared_flash_dispatch_calls =
            shared_flash_dispatch_calls;
        g_generation_flash_evidence.native_quantized_stage_count =
            native_quantized_stage_count;
        g_generation_flash_evidence.approved_dense_f32_stage_count =
            approved_dense_f32_stage_count;
        g_generation_flash_evidence.disallowed_fallback_stage_count =
            disallowed_fallback_stage_count;
        g_generation_flash_evidence.explicit_no_claim_stage_count =
            explicit_no_claim_stage_count;
        g_generation_flash_evidence.native_q8_0_dispatch_calls =
            native_q8_0_dispatch_calls;
        g_generation_flash_evidence.packed_q8_0_dispatch_calls =
            packed_q8_0_dispatch_calls;
        g_generation_flash_evidence.optimized_q2_dispatch_calls =
            optimized_q2_dispatch_calls;
        g_generation_flash_evidence.shared_q2_dispatch_calls =
            shared_q2_dispatch_calls;
        g_generation_flash_evidence.optimized_q3_dispatch_calls =
            optimized_q3_dispatch_calls;
        g_generation_flash_evidence.shared_q3_dispatch_calls =
            shared_q3_dispatch_calls;
        g_generation_flash_evidence.optimized_q4_dispatch_calls =
            optimized_q4_dispatch_calls;
        g_generation_flash_evidence.shared_q4_dispatch_calls =
            shared_q4_dispatch_calls;
        g_generation_flash_evidence.optimized_q6_dispatch_calls =
            optimized_q6_dispatch_calls;
        g_generation_flash_evidence.shared_q6_dispatch_calls =
            shared_q6_dispatch_calls;
        g_generation_flash_evidence.seam = seam;
      }
      if (generation_seam_audit_enabled()) {
        print_generation_seam_audit("emel", seam);
        verify_emel_generation_seam(seam);
      }
      static_cast<void>(sink);
    }
  }
}

void append_reference_generation_cases_for_current_benchmark_lane(
    std::vector<result> &results, const config &cfg) {
  const bool compare_lane =
      generation_lane_mode_current() == generation_lane_mode::compare;
  const config case_cfg = generation_case_config(cfg);

  if (compare_lane) {
    const auto &fixtures = maintained_compare_generation_fixtures();
    for (const prepared_generation_fixture &prepared_fixture : fixtures) {
      const generation_fixture_spec *spec = prepared_fixture.spec;
      const reference_fixture &fixture = prepared_fixture.reference;
      g_generation_fixture_rel = spec->fixture->fixture_rel;
      for (const generation_case_spec &generation_case :
           prepared_fixture.cases) {
        if (!generation_workload_selected(generation_case)) {
          continue;
        }
        volatile std::size_t sink = 0u;
        generation_seam_audit seam = {};
        generation_result latest_generated = {};
        validate_generation_formatter_contract(generation_case,
                                               fixture.formatter.contract);
        auto fn = [&]() {
          reset_generation_seam(seam);
          generation_result generated{};
          // Keep the reference path honest with EMEL's timed generate request:
          // formatting and tokenization stay inside the measured lambda here
          // too.
          if (!run_reference_generate(fixture, generation_case, seam,
                                      generated)) {
            fail_bench_setup("run_reference_generate",
                             generation_case.name.data());
          }
          latest_generated = generated;
          sink ^= generated.output_length;
        };

        const std::string case_name =
            generation_benchmark_case_name(generation_case.name);
        results.push_back(measure_case(case_name.c_str(), case_cfg, fn));
        result &compare_record = results.back();
        compare_record.compare_group = generation_case.manifest.compare_group;
        compare_record.benchmark_lane =
            std::string{generation_benchmark_lane_name()};
        compare_record.lane = "reference";
        compare_record.backend_id = "cpp.reference.llama_cpp";
        compare_record.backend_language = "cpp";
        compare_record.thread_count = generation_reference_thread_count();
        compare_record.thread_contract = generation_reference_thread_contract();
        compare_record.workload_id = generation_case.manifest.id;
        compare_record.workload_manifest_path =
            generation_case.manifest.workload_manifest_path;
        compare_record.comparison_mode =
            generation_case.manifest.comparison_mode;
        compare_record.model_id = generation_case.manifest.fixture_name;
        compare_record.fixture_id = generation_case.manifest.fixture_rel;
        compare_record.prompt_fixture_id =
            generation_case.manifest.prompt_fixture_id;
        compare_record.prompt_fixture_path =
            generation_case.manifest.prompt_fixture_path;
        compare_record.prompt_id = generation_case.manifest.prompt_id;
        compare_record.formatter_mode = generation_case.manifest.formatter_mode;
        compare_record.formatter_contract =
            generation_case.manifest.formatter_contract;
        compare_record.sampling_id = generation_case.manifest.sampling_id;
        compare_record.stop_id = generation_case.manifest.stop_id;
        compare_record.seed = generation_case.manifest.seed;
        compare_record.max_output_tokens =
            generation_case.manifest.max_output_tokens;
        compare_record.comparable = generation_case.manifest.comparable;
        capture_generation_output_metrics(compare_record, latest_generated);
        compare_record.note = generation_case.manifest.comparability_note;
        if (generation_seam_audit_enabled()) {
          print_generation_seam_audit("reference", seam);
          verify_reference_generation_seam(seam);
        }
        static_cast<void>(sink);
      }
    }
    return;
  }

  const auto &fixtures = maintained_reference_generation_fixtures();
  for (const prepared_reference_generation_fixture &prepared_fixture :
       fixtures) {
    const generation_fixture_spec *spec = prepared_fixture.spec;
    const reference_fixture &fixture = prepared_fixture.reference;
    g_generation_fixture_rel = spec->fixture->fixture_rel;
    for (const generation_case_spec &generation_case : prepared_fixture.cases) {
      if (!generation_workload_selected(generation_case)) {
        continue;
      }
      volatile std::size_t sink = 0u;
      generation_seam_audit seam = {};
      generation_result latest_generated = {};
      validate_generation_formatter_contract(generation_case,
                                             fixture.formatter.contract);
      auto fn = [&]() {
        reset_generation_seam(seam);
        generation_result generated{};
        // Keep the reference path honest with EMEL's timed generate request:
        // formatting and tokenization stay inside the measured lambda here too.
        if (!run_reference_generate(fixture, generation_case, seam,
                                    generated)) {
          fail_bench_setup("run_reference_generate",
                           generation_case.name.data());
        }
        latest_generated = generated;
        sink ^= generated.output_length;
      };

      const std::string case_name =
          generation_benchmark_case_name(generation_case.name);
      results.push_back(measure_case(case_name.c_str(), case_cfg, fn));
      result &compare_record = results.back();
      compare_record.compare_group = generation_case.manifest.compare_group;
      compare_record.benchmark_lane =
          std::string{generation_benchmark_lane_name()};
      compare_record.lane = "reference";
      compare_record.backend_id = "cpp.reference.llama_cpp";
      compare_record.backend_language = "cpp";
      compare_record.thread_count = generation_reference_thread_count();
      compare_record.thread_contract = generation_reference_thread_contract();
      compare_record.workload_id = generation_case.manifest.id;
      compare_record.workload_manifest_path =
          generation_case.manifest.workload_manifest_path;
      compare_record.comparison_mode = generation_case.manifest.comparison_mode;
      compare_record.model_id = generation_case.manifest.fixture_name;
      compare_record.fixture_id = generation_case.manifest.fixture_rel;
      compare_record.prompt_fixture_id =
          generation_case.manifest.prompt_fixture_id;
      compare_record.prompt_fixture_path =
          generation_case.manifest.prompt_fixture_path;
      compare_record.prompt_id = generation_case.manifest.prompt_id;
      compare_record.formatter_mode = generation_case.manifest.formatter_mode;
      compare_record.formatter_contract =
          generation_case.manifest.formatter_contract;
      compare_record.sampling_id = generation_case.manifest.sampling_id;
      compare_record.stop_id = generation_case.manifest.stop_id;
      compare_record.seed = generation_case.manifest.seed;
      compare_record.max_output_tokens =
          generation_case.manifest.max_output_tokens;
      compare_record.comparable = generation_case.manifest.comparable;
      capture_generation_output_metrics(compare_record, latest_generated);
      compare_record.note = generation_case.manifest.comparability_note;
      if (generation_seam_audit_enabled()) {
        print_generation_seam_audit("reference", seam);
        verify_reference_generation_seam(seam);
      }
      static_cast<void>(sink);
    }
  }
}

void append_emel_generation_cases(std::vector<result> &results,
                                  const config &cfg) {
  const generation_benchmark_lane_selection selection =
      selected_generation_benchmark_lanes();
  reset_generation_flash_evidence();
  for (size_t index = 0u; index < selection.count; ++index) {
    const scoped_generation_benchmark_lane lane_scope{selection.lanes[index]};
    append_emel_generation_cases_for_current_benchmark_lane(results, cfg);
  }
}

void append_reference_generation_cases(std::vector<result> &results,
                                       const config &cfg) {
  const generation_benchmark_lane_selection selection =
      selected_generation_benchmark_lanes();
  for (size_t index = 0u; index < selection.count; ++index) {
    const scoped_generation_benchmark_lane lane_scope{selection.lanes[index]};
    append_reference_generation_cases_for_current_benchmark_lane(results, cfg);
  }
}

} // namespace emel::bench
