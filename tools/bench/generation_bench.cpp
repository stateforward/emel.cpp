#include "bench_cases.hpp"
#include "generation_compare_contract.hpp"
#include "generation_workload_manifest.hpp"
#include "embedding_generator_bench_helpers.hpp"
#include "../generation_formatter_contract.hpp"
#include "../generation_fixture_registry.hpp"

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
#include <utility>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/errors.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/generator/errors.hpp"
#include "emel/generator/events.hpp"
#include "emel/generator/context.hpp"
#include "emel/generator/detail.hpp"
#include "emel/generator/prefill/context.hpp"
#include "emel/generator/prefill/guards.hpp"
#include "emel/generator/sm.hpp"
#include "emel/logits/sampler/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/weight_loader/errors.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/renderer/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

#include "ggml.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-memory.h"
#include "llama-vocab.h"

namespace {

constexpr size_t k_generation_output_capacity = 65536u;

struct generation_case_spec {
  std::string name = {};
  std::string prompt = {};
  int32_t max_tokens = 0;
  emel::bench::generation_workload_manifest manifest = {};
};

struct generation_fixture_spec {
  const emel::tools::generation_fixture_registry::maintained_fixture * fixture = nullptr;
};

constexpr emel::tools::generation_fixture_registry::maintained_fixture
    k_gemma4_emel_generation_fixture = {
        .name = "gemma-4-e2b-it-Q8_0.gguf",
        .slug = "gemma_4_e2b_it_q8_0",
        .fixture_rel = "tests/models/gemma-4-e2b-it-Q8_0.gguf",
        .current_publication = false,
    };

constexpr generation_fixture_spec k_qwen3_generation_fixture = {
    .fixture = &emel::tools::generation_fixture_registry::k_qwen3_generation_fixture,
};

constexpr generation_fixture_spec k_lfm2_generation_fixture = {
    .fixture = &emel::tools::generation_fixture_registry::k_lfm2_generation_fixture,
};

constexpr generation_fixture_spec k_gemma4_generation_fixture = {
    .fixture = &k_gemma4_emel_generation_fixture,
};

constexpr std::array<generation_fixture_spec, 2> k_compare_generation_fixtures = {
    k_qwen3_generation_fixture,
    k_lfm2_generation_fixture,
};

constexpr std::array<generation_fixture_spec, 3> k_emel_generation_fixtures = {
    k_qwen3_generation_fixture,
    k_lfm2_generation_fixture,
    k_gemma4_generation_fixture,
};

using llama_model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using llama_context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

constexpr llama_flash_attn_type k_reference_flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

std::uint64_t read_env_u64(const char * name, const std::uint64_t fallback) {
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

std::size_t read_env_size(const char * name, const std::size_t fallback) {
  const auto parsed = read_env_u64(name, static_cast<std::uint64_t>(fallback));
  return parsed == 0u ? fallback : static_cast<std::size_t>(parsed);
}

bool env_enabled(const char * name) {
  const char * value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

std::string_view generation_workload_filter() {
  const char * value = std::getenv("EMEL_GENERATION_WORKLOAD_ID");
  if (value == nullptr || value[0] == '\0') {
    return {};
  }
  return value;
}

bool generation_workload_selected(const generation_case_spec & spec) {
  const std::string_view filter = generation_workload_filter();
  if (filter.empty()) {
    return true;
  }
  return spec.manifest.id == filter || spec.name == filter ||
      spec.manifest.compare_group == filter;
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
    const emel::tools::generation_fixture_registry::maintained_fixture & fixture) {
  return bench_root_path() / fixture.fixture_rel;
}

bool generation_fixture_exists(
    const emel::tools::generation_fixture_registry::maintained_fixture & fixture) {
  return std::filesystem::exists(generation_fixture_path(fixture));
}

void report_missing_generation_fixture(
    const emel::tools::generation_fixture_registry::maintained_fixture & fixture) {
  std::fprintf(stderr,
               "warning: skipping missing generation fixture %.*s (%.*s)\n",
               static_cast<int>(fixture.name.size()),
               fixture.name.data(),
               static_cast<int>(fixture.fixture_rel.size()),
               fixture.fixture_rel.data());
}

extern std::string g_generation_formatter_contract;
extern std::string g_generation_architecture_contract;
extern std::string_view g_generation_fixture_rel;

[[noreturn]] void fail_bench_setup(const char * step, const char * detail) {
  const std::string_view fixture_rel =
      g_generation_fixture_rel.empty() ? std::string_view{"<unknown>"} : g_generation_fixture_rel;
  std::fprintf(stderr,
               "# generation_fixture: %.*s\n",
               static_cast<int>(fixture_rel.size()),
               fixture_rel.data());
  if (!g_generation_architecture_contract.empty()) {
    std::fprintf(stderr,
                 "# generation_architecture: %.*s\n",
                 static_cast<int>(g_generation_architecture_contract.size()),
                 g_generation_architecture_contract.data());
  }
  if (!g_generation_formatter_contract.empty()) {
    std::fprintf(stderr,
                 "# generation_formatter_contract: %.*s\n",
                 static_cast<int>(g_generation_formatter_contract.size()),
                 g_generation_formatter_contract.data());
  }
  std::fprintf(stderr, "error: generation bench setup failed at %s (%s)\n", step, detail);
  std::exit(1);
}

generation_case_spec load_generation_case_spec(
    emel::bench::generation_workload_manifest manifest) {
  generation_case_spec spec = {};
  spec.manifest = std::move(manifest);
  spec.name = spec.manifest.case_name;
  spec.prompt = spec.manifest.prompt_text;
  spec.max_tokens = static_cast<int32_t>(spec.manifest.max_output_tokens);
  return spec;
}

const std::vector<generation_case_spec> & maintained_generation_workloads() {
  static const std::vector<generation_case_spec> workloads = [] {
    std::string error = {};
    std::vector<emel::bench::generation_workload_manifest> manifests = {};
    if (!emel::bench::load_generation_workload_manifests(bench_root_path(), manifests, &error)) {
      fail_bench_setup("load_generation_workload_manifests", error.c_str());
    }
    std::vector<generation_case_spec> loaded = {};
    loaded.reserve(manifests.size());
    for (auto & manifest : manifests) {
      loaded.push_back(load_generation_case_spec(std::move(manifest)));
    }
    return loaded;
  }();
  return workloads;
}

std::vector<generation_case_spec> generation_cases_for_fixture(
    const emel::tools::generation_fixture_registry::maintained_fixture & fixture,
    const bool comparable_only) {
  std::vector<generation_case_spec> cases = {};
  for (const generation_case_spec & candidate : maintained_generation_workloads()) {
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
    const emel::tools::generation_fixture_registry::maintained_fixture & fixture,
    const generation_case_spec & spec) {
  if (spec.manifest.fixture_rel != fixture.fixture_rel ||
      spec.manifest.fixture_slug != fixture.slug ||
      spec.manifest.fixture_name != fixture.name) {
    fail_bench_setup("validate_generation_workload_fixture", spec.manifest.id.c_str());
  }
}

void validate_generation_formatter_contract(const generation_case_spec & spec,
                                            const std::string_view actual_contract) {
  if (spec.manifest.formatter_contract != actual_contract) {
    fail_bench_setup("validate_generation_formatter_contract", spec.manifest.id.c_str());
  }
}

struct llama_backend_guard {
  llama_backend_guard() { llama_backend_init(); }
  ~llama_backend_guard() { llama_backend_free(); }
};

void silence_llama_log(ggml_log_level, const char *, void *) {}

struct llama_log_silencer {
  ggml_log_callback callback = nullptr;
  void * user_data = nullptr;

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
void copy_name(std::array<char, k_array_size> & dst, const std::string_view value) {
  static_assert(k_array_size > 0, "copy_name requires non-empty destination");
  dst.fill('\0');
  const size_t copy_len = std::min(value.size(), k_array_size - 1);
  if (copy_len > 0u) {
    std::memcpy(dst.data(), value.data(), copy_len);
  }
}

emel::text::tokenizer::preprocessor::preprocessor_kind generation_preprocessor_variant(
    const emel::model::data & model_data) {
  using preprocessor_kind = emel::text::tokenizer::preprocessor::preprocessor_kind;
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

emel::text::encoders::encoder_kind generation_encoder_variant(
    const emel::model::data & model_data) {
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

std::string_view vocab_token_view(const emel::model::data::vocab & vocab, const int32_t token_id) {
  if (token_id < 0 || static_cast<uint32_t>(token_id) >= vocab.n_tokens) {
    return {};
  }

  const auto & entry = vocab.entries[static_cast<size_t>(token_id)];
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

struct weight_capture {
  bool bind_done = false;
  bool bind_error = false;
  bool plan_done = false;
  bool plan_error = false;
  bool apply_done = false;
  bool apply_error = false;
  uint32_t effect_count = 0u;
  emel::error::type err = emel::error::cast(emel::model::weight_loader::error::none);
};

struct load_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
  uint64_t bytes_total = 0u;
  uint64_t bytes_done = 0u;
  bool used_mmap = false;
};

struct initialize_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::generator::error::none);
};

struct generation_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::generator::error::none);
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

struct generation_result {
  std::array<char, k_generation_output_capacity> output = {};
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

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
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::vector<emel::model::weight_loader::effect_request> effect_requests = {};
  std::vector<emel::model::weight_loader::effect_result> effect_results = {};
  emel::gguf::loader::sm gguf_loader = {};
  emel::model::weight_loader::sm weight_loader = {};
  emel::model::loader::sm model_loader = {};
  gguf_capture gguf = {};
  weight_capture weight = {};
  load_capture load = {};
  emel::tools::generation_formatter_contract::formatter_binding formatter_binding = {};
};

struct reference_fixture {
  llama_model_ptr model = {nullptr, llama_model_free};
  llama_context_ptr context = {nullptr, llama_free};
  const llama_vocab * vocab = nullptr;
  int32_t vocab_size = 0;
  emel::tools::generation_formatter_contract::reference_formatter_info formatter = {};
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
  emel::text::tokenizer::sm tokenizer = {};
  emel::text::conditioner::sm conditioner = {};
  std::unique_ptr<emel::generator::sm> generator = {};
  std::array<emel::logits::sampler::fn, 1> samplers = {};
  emel::tools::generation_formatter_contract::formatter_binding formatter_binding = {};
  generation_seam_audit seam = {};
  initialize_capture initialize = {};
  generation_capture generation = {};
};

struct prepared_generation_fixture {
  const generation_fixture_spec * spec = nullptr;
  std::vector<generation_case_spec> cases = {};
  emel_fixture emel = {};
  reference_fixture reference = {};
};

struct prepared_emel_generation_fixture {
  const generation_fixture_spec * spec = nullptr;
  std::vector<generation_case_spec> cases = {};
  emel_fixture emel = {};
};

struct prepared_reference_generation_fixture {
  const generation_fixture_spec * spec = nullptr;
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
    case emel::model::loader::error::untracked:
      return "untracked";
  }
  return "unknown";
}

emel::model::detail::kv_binding kv_binding_from_fixture(const emel_fixture & fixture) {
  return emel::model::detail::kv_binding{
      .arena = std::span<const uint8_t>{fixture.kv_arena.data(), fixture.kv_arena.size()},
      .entries = std::span<const emel::gguf::loader::kv_entry>{fixture.kv_entries.data(),
                                                               fixture.kv_entries.size()},
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

bool read_file_bytes(const std::string & path, std::vector<uint8_t> & out) {
  out.clear();

  std::FILE * file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }

  const bool seek_end_ok = std::fseek(file, 0, SEEK_END) == 0;
  const long file_size = seek_end_ok ? std::ftell(file) : -1L;
  const bool seek_start_ok = file_size >= 0L && std::fseek(file, 0, SEEK_SET) == 0;
  if (!seek_end_ok || file_size < 0L || !seek_start_ok) {
    std::fclose(file);
    return false;
  }

  out.resize(static_cast<size_t>(file_size));
  const size_t read_size = out.empty() ? 0u : std::fread(out.data(), 1u, out.size(), file);
  std::fclose(file);
  return read_size == out.size();
}

emel::error::type sampler_select_argmax(int32_t & candidate_ids,
                                        float & candidate_scores,
                                        int32_t & candidate_count,
                                        int32_t & selected_token_out) {
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

template <class fixture_type>
void reset_gguf_capture(fixture_type & fixture) {
  fixture.gguf = {};
}

void reset_weight_capture(emel_fixture & fixture) { fixture.weight = {}; }
void reset_load_capture(emel_fixture & fixture) { fixture.load = {}; }
void reset_initialize_capture(emel_session & session) { session.initialize = {}; }
void reset_generation_capture(emel_session & session) { session.generation = {}; }

template <class fixture_type>
void on_probe_done_impl(void * owner, const emel::gguf::loader::events::probe_done & ev) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.probe_done = true;
  fixture.gguf.probe_error = false;
  fixture.gguf.requirements = ev.requirements_out;
}

void on_probe_done(void * owner, const emel::gguf::loader::events::probe_done & ev) {
  on_probe_done_impl<emel_fixture>(owner, ev);
}

template <class fixture_type>
void on_probe_error_impl(void * owner, const emel::gguf::loader::events::probe_error & ev) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.probe_error = true;
  fixture.gguf.err = ev.err;
}

void on_probe_error(void * owner, const emel::gguf::loader::events::probe_error & ev) {
  on_probe_error_impl<emel_fixture>(owner, ev);
}

template <class fixture_type>
void on_bind_done_impl(void * owner, const emel::gguf::loader::events::bind_done &) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.bind_done = true;
  fixture.gguf.bind_error = false;
}

void on_bind_done(void * owner, const emel::gguf::loader::events::bind_done & ev) {
  on_bind_done_impl<emel_fixture>(owner, ev);
}

template <class fixture_type>
void on_bind_error_impl(void * owner, const emel::gguf::loader::events::bind_error & ev) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.bind_error = true;
  fixture.gguf.err = ev.err;
}

void on_bind_error(void * owner, const emel::gguf::loader::events::bind_error & ev) {
  on_bind_error_impl<emel_fixture>(owner, ev);
}

template <class fixture_type>
void on_parse_done_impl(void * owner, const emel::gguf::loader::events::parse_done &) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.parse_done = true;
  fixture.gguf.parse_error = false;
}

void on_parse_done(void * owner, const emel::gguf::loader::events::parse_done & ev) {
  on_parse_done_impl<emel_fixture>(owner, ev);
}

template <class fixture_type>
void on_parse_error_impl(void * owner, const emel::gguf::loader::events::parse_error & ev) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.parse_error = true;
  fixture.gguf.err = ev.err;
}

void on_parse_error(void * owner, const emel::gguf::loader::events::parse_error & ev) {
  on_parse_error_impl<emel_fixture>(owner, ev);
}

void on_weight_bind_done(void * owner, const emel::model::weight_loader::events::bind_done &) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.bind_done = true;
  fixture.weight.bind_error = false;
}

void on_weight_bind_error(void * owner,
                          const emel::model::weight_loader::events::bind_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.bind_error = true;
  fixture.weight.err = ev.err;
}

void on_weight_plan_done(void * owner,
                         const emel::model::weight_loader::events::plan_done & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.plan_done = true;
  fixture.weight.plan_error = false;
  fixture.weight.effect_count = ev.effect_count;
}

void on_weight_plan_error(void * owner,
                          const emel::model::weight_loader::events::plan_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.plan_error = true;
  fixture.weight.err = ev.err;
}

void on_weight_apply_done(void * owner, const emel::model::weight_loader::events::apply_done &) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.apply_done = true;
  fixture.weight.apply_error = false;
}

void on_weight_apply_error(void * owner,
                           const emel::model::weight_loader::events::apply_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.apply_error = true;
  fixture.weight.err = ev.err;
}

void on_load_done(void * owner, const emel::model::loader::events::load_done & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.load.done = true;
  fixture.load.error = false;
  fixture.load.err = emel::error::cast(emel::model::loader::error::none);
  fixture.load.bytes_total = ev.bytes_total;
  fixture.load.bytes_done = ev.bytes_done;
  fixture.load.used_mmap = ev.used_mmap;
}

void on_load_error(void * owner, const emel::model::loader::events::load_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.load.error = true;
  fixture.load.err = ev.err;
}

void on_initialize_done(void * owner, const emel::generator::events::initialize_done &) {
  auto & session = *static_cast<emel_session *>(owner);
  session.initialize.done = true;
  session.initialize.error = false;
  session.initialize.err = emel::error::cast(emel::generator::error::none);
}

void on_initialize_error(void * owner, const emel::generator::events::initialize_error & ev) {
  auto & session = *static_cast<emel_session *>(owner);
  session.initialize.error = true;
  session.initialize.err = ev.err;
}

void on_generation_done(void * owner, const emel::generator::events::generation_done & ev) {
  auto & session = *static_cast<emel_session *>(owner);
  session.generation.done = true;
  session.generation.error = false;
  session.generation.err = emel::error::cast(emel::generator::error::none);
  session.generation.tokens_generated = ev.tokens_generated;
  session.generation.output_length = ev.output_length;
}

void on_generation_error(void * owner, const emel::generator::events::generation_error & ev) {
  auto & session = *static_cast<emel_session *>(owner);
  session.generation.error = true;
  session.generation.err = ev.err;
  session.generation.tokens_generated = ev.tokens_generated;
  session.generation.output_length = ev.output_length;
}

bool tokenizer_bind_dispatch(void * tokenizer_sm,
                             const emel::text::tokenizer::event::bind & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

bool tokenizer_tokenize_dispatch(void * tokenizer_sm,
                                 const emel::text::tokenizer::event::tokenize & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

void reset_generation_seam(generation_seam_audit & seam) {
  seam = {};
}

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

void print_generation_seam_audit(const char * label, const generation_seam_audit & seam) {
  std::fprintf(stderr,
               "generation_bench_seams[%s]: emel_decode_calls=%d emel_logits_calls=%d "
               "emel_formatter_calls=%d emel_tokenize_calls=%d emel_vocab_calls=%d "
               "reference_decode_calls=%d reference_logits_calls=%d "
               "reference_formatter_calls=%d reference_tokenize_calls=%d "
               "reference_vocab_calls=%d\n",
               label,
               seam.emel_reference_decode_calls,
               seam.emel_reference_logits_calls,
               seam.emel_reference_formatter_calls,
               seam.emel_reference_tokenize_calls,
               seam.emel_reference_vocab_calls,
               seam.direct_reference_decode_calls,
               seam.direct_reference_logits_calls,
               seam.direct_reference_formatter_calls,
               seam.direct_reference_tokenize_calls,
               seam.direct_reference_vocab_calls);
}

void verify_emel_generation_seam(const generation_seam_audit & seam) {
  if (seam.emel_reference_decode_calls != 0 || seam.emel_reference_logits_calls != 0 ||
      seam.emel_reference_formatter_calls != 0 || seam.emel_reference_tokenize_calls != 0 ||
      seam.emel_reference_vocab_calls != 0 || seam.direct_reference_decode_calls != 0 ||
      seam.direct_reference_logits_calls != 0 || seam.direct_reference_formatter_calls != 0 ||
      seam.direct_reference_tokenize_calls != 0 || seam.direct_reference_vocab_calls != 0) {
    fail_bench_setup("generation seam audit", "EMEL benchmark path touched reference decode seam");
  }
}

std::string_view generator_error_name(const emel::error::type err) noexcept {
  switch (static_cast<emel::generator::error>(err)) {
    case emel::generator::error::none:
      return "none";
    case emel::generator::error::invalid_request:
      return "invalid_request";
    case emel::generator::error::backend:
      return "backend";
  }
  return "unknown";
}

void verify_reference_generation_seam(const generation_seam_audit & seam) {
  if (seam.emel_reference_decode_calls != 0 || seam.emel_reference_logits_calls != 0 ||
      seam.emel_reference_formatter_calls != 0 || seam.emel_reference_tokenize_calls != 0 ||
      seam.emel_reference_vocab_calls != 0 || seam.direct_reference_decode_calls <= 0 ||
      seam.direct_reference_logits_calls <= 0 || seam.direct_reference_formatter_calls <= 0 ||
      seam.direct_reference_tokenize_calls <= 0 || seam.direct_reference_vocab_calls <= 0) {
    fail_bench_setup("generation seam audit",
                     "reference benchmark path did not stay on the explicit reference seam");
  }
}

int32_t run_direct_reference_decode(generation_seam_audit & seam,
                                    llama_context * ctx,
                                    const llama_batch batch) {
  seam.direct_reference_decode_calls += 1;
  return llama_decode(ctx, batch);
}

float * read_direct_reference_logits(generation_seam_audit & seam, llama_context * ctx) {
  seam.direct_reference_logits_calls += 1;
  return llama_get_logits_ith(ctx, -1);
}

bool format_direct_reference_prompt(
    generation_seam_audit & seam,
    const emel::tools::generation_formatter_contract::reference_formatter_info & formatter,
    const std::string_view prompt,
    std::string & formatted_prompt) {
  seam.direct_reference_formatter_calls += 1;
  return emel::tools::generation_formatter_contract::format_reference_single_user_prompt(
      formatter,
      prompt,
      formatted_prompt);
}

int32_t tokenize_direct_reference_prompt(generation_seam_audit & seam,
                                         const llama_vocab * vocab,
                                         const std::string & formatted_prompt,
                                         llama_token * tokens,
                                         const int32_t token_capacity) {
  seam.direct_reference_tokenize_calls += 1;
  return llama_tokenize(vocab,
                        formatted_prompt.data(),
                        static_cast<int32_t>(formatted_prompt.size()),
                        tokens,
                        token_capacity,
                        false,
                        false);
}

bool reference_vocab_is_control(generation_seam_audit & seam,
                                const llama_vocab * vocab,
                                const llama_token token) {
  seam.direct_reference_vocab_calls += 1;
  return llama_vocab_is_control(vocab, token);
}

bool reference_vocab_is_eog(generation_seam_audit & seam,
                            const llama_vocab * vocab,
                            const llama_token token) {
  seam.direct_reference_vocab_calls += 1;
  return llama_vocab_is_eog(vocab, token);
}

const char * reference_vocab_text(generation_seam_audit & seam,
                                  const llama_vocab * vocab,
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

emel::error::type map_weight_loader_error(const emel::error::type err) {
  using model_error = emel::model::loader::error;
  using weight_error = emel::model::weight_loader::error;

  switch (err) {
    case emel::error::cast(weight_error::none):
      return emel::error::cast(model_error::none);
    case emel::error::cast(weight_error::invalid_request):
      return emel::error::cast(model_error::invalid_request);
    case emel::error::cast(weight_error::capacity):
    case emel::error::cast(weight_error::backend_error):
    case emel::error::cast(weight_error::out_of_memory):
      return emel::error::cast(model_error::backend_error);
    case emel::error::cast(weight_error::model_invalid):
      return emel::error::cast(model_error::model_invalid);
    case emel::error::cast(weight_error::internal_error):
      return emel::error::cast(model_error::internal_error);
    case emel::error::cast(weight_error::untracked):
    default:
      return emel::error::cast(model_error::untracked);
  }
}

template <class fixture_type>
std::string_view kv_key_view(const fixture_type & fixture,
                             const emel::gguf::loader::kv_entry & entry) {
  if (static_cast<size_t>(entry.key_offset) + static_cast<size_t>(entry.key_length) >
      fixture.kv_arena.size()) {
    return {};
  }

  return std::string_view{
      reinterpret_cast<const char *>(fixture.kv_arena.data() + entry.key_offset),
      entry.key_length,
  };
}

template <class fixture_type>
std::span<const uint8_t> kv_value_view(const fixture_type & fixture,
                                       const emel::gguf::loader::kv_entry & entry) {
  if (static_cast<size_t>(entry.value_offset) + static_cast<size_t>(entry.value_length) >
      fixture.kv_arena.size()) {
    return {};
  }

  return std::span<const uint8_t>{fixture.kv_arena.data() + entry.value_offset, entry.value_length};
}

template <class fixture_type>
const emel::gguf::loader::kv_entry * find_kv_entry(const fixture_type & fixture,
                                                   const std::string_view key) {
  for (const auto & entry : fixture.kv_entries) {
    if (kv_key_view(fixture, entry) == key) {
      return &entry;
    }
  }
  return nullptr;
}

template <class fixture_type>
bool decode_string_value(const fixture_type & fixture,
                         const emel::gguf::loader::kv_entry & entry,
                         std::string_view & value_out);

template <class fixture_type>
emel::tools::generation_formatter_contract::formatter_binding
resolve_fixture_formatter_binding(const fixture_type & fixture) {
  std::string_view primary_template = {};
  const auto * entry = find_kv_entry(fixture, "tokenizer.chat_template");
  if (entry != nullptr && !decode_string_value(fixture, *entry, primary_template)) {
    return emel::tools::generation_formatter_contract::formatter_binding{
        .formatter_ctx = nullptr,
        .format_prompt = emel::text::formatter::format_raw,
        .support = emel::tools::generation_formatter_contract::support_kind::unsupported_template,
        .contract = emel::tools::generation_formatter_contract::k_unsupported_template_contract,
    };
  }

  uint32_t named_template_count = 0u;
  for (const auto & candidate : fixture.kv_entries) {
    const std::string_view key = kv_key_view(fixture, candidate);
    if (key.starts_with("tokenizer.chat_template.") &&
        key != "tokenizer.chat_template") {
      named_template_count += 1u;
    }
  }

  return emel::tools::generation_formatter_contract::resolve_primary_template_binding(
      primary_template,
      named_template_count);
}

template <class fixture_type>
bool decode_integer_value(const fixture_type & fixture,
                          const emel::gguf::loader::kv_entry & entry,
                          uint64_t & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::detail::constants;

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
bool decode_float_value(const fixture_type & fixture,
                        const emel::gguf::loader::kv_entry & entry,
                        float & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::detail::constants;

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
bool decode_bool_value(const fixture_type & fixture,
                       const emel::gguf::loader::kv_entry & entry,
                       bool & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::detail::constants;
  if (entry.value_type != constants::gguf_type_bool || bytes.size() != 1u) {
    return false;
  }

  value_out = bytes[0] != 0u;
  return true;
}

template <class fixture_type>
bool decode_string_value(const fixture_type & fixture,
                         const emel::gguf::loader::kv_entry & entry,
                         std::string_view & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  if (entry.value_type != constants::gguf_type_string || bytes.size() < sizeof(uint64_t)) {
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
bool decode_string_array_count(const fixture_type & fixture,
                               const emel::gguf::loader::kv_entry & entry,
                               uint32_t & count_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  if (entry.value_type != constants::gguf_type_array ||
      bytes.size() < sizeof(uint32_t) + sizeof(uint64_t)) {
    return false;
  }

  const uint32_t element_type = read_u32_le(bytes.first(sizeof(uint32_t)));
  if (element_type != constants::gguf_type_string) {
    return false;
  }

  const uint64_t count = read_u64_le(bytes.subspan(sizeof(uint32_t), sizeof(uint64_t)));
  if (count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  count_out = static_cast<uint32_t>(count);
  return true;
}

template <class fixture_type>
bool decode_integer_array_first_nonzero(const fixture_type & fixture,
                                        const emel::gguf::loader::kv_entry & entry,
                                        int32_t & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  if (entry.value_type != constants::gguf_type_array ||
      bytes.size() < sizeof(uint32_t) + sizeof(uint64_t)) {
    return false;
  }

  const uint32_t element_type = read_u32_le(bytes.first(sizeof(uint32_t)));
  const uint64_t count = read_u64_le(bytes.subspan(sizeof(uint32_t), sizeof(uint64_t)));
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
    const std::span<const uint8_t> element =
        payload.subspan(static_cast<size_t>(index * element_size), element_size);
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
        raw_value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      continue;
    }

    value_out = static_cast<int32_t>(raw_value);
    return true;
  }

  return false;
}

bool copy_tensor_names(const std::span<const uint8_t> file_image, emel::model::data & model_data) {
  model_data.name_bytes_used = 0u;

  for (uint32_t i = 0u; i < model_data.n_tensors; ++i) {
    auto & tensor = model_data.tensors[i];
    const size_t name_offset = static_cast<size_t>(tensor.name_offset);
    const size_t name_length = static_cast<size_t>(tensor.name_length);
    if (name_offset + name_length > file_image.size() ||
        model_data.name_bytes_used + name_length > model_data.name_storage.size()) {
      return false;
    }

    const uint32_t copied_offset = model_data.name_bytes_used;
    if (name_length > 0u) {
      std::memcpy(model_data.name_storage.data() + copied_offset,
                  file_image.data() + name_offset,
                  name_length);
    }

    model_data.name_bytes_used += static_cast<uint32_t>(name_length);
    tensor.name_offset = copied_offset;
  }
  return true;
}

std::string_view tensor_name_view(const emel::model::data & model_data,
                                  const emel::model::data::tensor_record & tensor) {
  if (static_cast<size_t>(tensor.name_offset) + static_cast<size_t>(tensor.name_length) >
      model_data.name_storage.size()) {
    return {};
  }

  return std::string_view{model_data.name_storage.data() + tensor.name_offset, tensor.name_length};
}

bool try_parse_block_index(const std::string_view name, int32_t & block_index_out) {
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

emel::error::type populate_model_metadata(const emel_fixture & fixture,
                                          emel::model::data & model_data) {
  return emel::model::detail::load_hparams_from_gguf(kv_binding_from_fixture(fixture), model_data)
             ? emel::error::cast(emel::model::loader::error::none)
             : emel::error::cast(emel::model::loader::error::model_invalid);
}

std::string_view architecture_name_view(const emel::model::data & model_data) {
  size_t length = 0u;
  while (length < model_data.architecture_name.size() &&
         model_data.architecture_name[length] != '\0') {
    ++length;
  }
  return std::string_view{model_data.architecture_name.data(), length};
}

emel::error::type run_emel_parse_model(void * owner,
                                       const emel::model::loader::event::load & req) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  if (req.file_image == nullptr || req.file_size == 0u) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }

  const std::span<const uint8_t> file_image{
      static_cast<const uint8_t *>(req.file_image),
      static_cast<size_t>(req.file_size),
  };

  reset_gguf_capture(fixture);
  emel::gguf::loader::requirements requirements = {};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb{&fixture, on_probe_done};
  const emel::gguf::loader::event::probe_error_fn probe_error_cb{&fixture, on_probe_error};
  const emel::gguf::loader::event::probe probe_ev{
      file_image,
      requirements,
      probe_done_cb,
      probe_error_cb,
  };
  if (!fixture.gguf_loader.process_event(probe_ev) || !fixture.gguf.probe_done ||
      fixture.gguf.probe_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  if (requirements.tensor_count > static_cast<uint32_t>(emel::model::data::k_max_tensors)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const uint64_t arena_bytes =
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements);
  if (arena_bytes == std::numeric_limits<uint64_t>::max()) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }

  fixture.kv_arena.resize(static_cast<size_t>(arena_bytes));
  fixture.kv_entries.resize(requirements.kv_count);

  reset_gguf_capture(fixture);
  const emel::gguf::loader::event::bind_done_fn bind_done_cb{&fixture, on_bind_done};
  const emel::gguf::loader::event::bind_error_fn bind_error_cb{&fixture, on_bind_error};
  const emel::gguf::loader::event::bind_storage bind_ev{
      std::span<uint8_t>{fixture.kv_arena},
      std::span<emel::gguf::loader::kv_entry>{fixture.kv_entries},
      std::span<emel::model::data::tensor_record>{req.model_data.tensors.data(),
                                                  requirements.tensor_count},
      bind_done_cb,
      bind_error_cb,
  };
  if (!fixture.gguf_loader.process_event(bind_ev) || !fixture.gguf.bind_done ||
      fixture.gguf.bind_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  reset_gguf_capture(fixture);
  const emel::gguf::loader::event::parse_done_fn parse_done_cb{&fixture, on_parse_done};
  const emel::gguf::loader::event::parse_error_fn parse_error_cb{&fixture, on_parse_error};
  const emel::gguf::loader::event::parse parse_ev{
      file_image,
      parse_done_cb,
      parse_error_cb,
  };
  if (!fixture.gguf_loader.process_event(parse_ev) || !fixture.gguf.parse_done ||
      fixture.gguf.parse_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  req.model_data.n_tensors = requirements.tensor_count;
  if (!copy_tensor_names(file_image, req.model_data)) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }

  return populate_model_metadata(fixture, req.model_data);
}

emel::error::type run_emel_load_weights(void * owner,
                                        const emel::model::loader::event::load & req,
                                        uint64_t & bytes_total,
                                        uint64_t & bytes_done,
                                        bool & used_mmap) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  if (req.model_data.n_tensors == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  fixture.effect_requests.resize(req.model_data.n_tensors);
  fixture.effect_results.resize(req.model_data.n_tensors);

  reset_weight_capture(fixture);
  emel::model::weight_loader::event::bind_storage bind_ev{
      std::span<emel::model::data::tensor_record>{req.model_data.tensors.data(),
                                                  req.model_data.n_tensors},
  };
  bind_ev.on_done = {&fixture, on_weight_bind_done};
  bind_ev.on_error = {&fixture, on_weight_bind_error};
  if (!fixture.weight_loader.process_event(bind_ev) || !fixture.weight.bind_done ||
      fixture.weight.bind_error) {
    return map_weight_loader_error(fixture.weight.err);
  }

  reset_weight_capture(fixture);
  emel::model::weight_loader::event::plan_load plan_ev{
      std::span<emel::model::weight_loader::effect_request>{fixture.effect_requests},
  };
  plan_ev.on_done = {&fixture, on_weight_plan_done};
  plan_ev.on_error = {&fixture, on_weight_plan_error};
  if (!fixture.weight_loader.process_event(plan_ev) || !fixture.weight.plan_done ||
      fixture.weight.plan_error) {
    return map_weight_loader_error(fixture.weight.err);
  }

  const uint32_t effect_count = fixture.weight.effect_count;
  for (uint32_t i = 0u; i < effect_count; ++i) {
    fixture.effect_results[i] = emel::model::weight_loader::effect_result{
        .kind = fixture.effect_requests[i].kind,
        .handle = fixture.effect_requests[i].target,
        .err = emel::error::cast(emel::model::weight_loader::error::none),
    };
  }

  reset_weight_capture(fixture);
  emel::model::weight_loader::event::apply_effect_results apply_ev{
      std::span<const emel::model::weight_loader::effect_result>{fixture.effect_results.data(),
                                                                 effect_count},
  };
  apply_ev.on_done = {&fixture, on_weight_apply_done};
  apply_ev.on_error = {&fixture, on_weight_apply_error};
  if (!fixture.weight_loader.process_event(apply_ev) || !fixture.weight.apply_done ||
      fixture.weight.apply_error) {
    return map_weight_loader_error(fixture.weight.err);
  }

  req.model_data.weights_data = req.file_image;
  req.model_data.weights_size = req.file_size;
  req.model_data.weights_mapped = false;
  req.model_data.weights_split_count = 1u;
  req.model_data.weights_split_offsets[0] = 0u;
  req.model_data.weights_split_sizes[0] = req.file_size;
  bytes_total = req.file_size;
  bytes_done = req.file_size;
  used_mmap = false;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type run_emel_map_layers(void *, const emel::model::loader::event::load & req) {
  int32_t max_block_index = -1;
  for (uint32_t i = 0u; i < req.model_data.n_tensors; ++i) {
    int32_t block_index = -1;
    if (emel::model::try_parse_block_index(
            emel::model::tensor_name_view(req.model_data, req.model_data.tensors[i]),
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

emel::error::type run_emel_validate_structure(void *,
                                              const emel::model::loader::event::load & req) {
  if (req.model_data.n_tensors == 0u || req.model_data.n_layers <= 0 ||
      req.model_data.weights_data == nullptr || req.model_data.weights_size == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type run_emel_validate_architecture(void *,
                                                 const emel::model::loader::event::load & req) {
  return emel::model::validate_execution_contract(req.model_data);
}

bool prepare_emel_fixture(emel_fixture & fixture, const std::string & model_path) {
  if (!read_file_bytes(model_path, fixture.file_bytes)) {
    return false;
  }

  reset_load_capture(fixture);
  emel::model::loader::event::parse_model_fn parse_model{&fixture, run_emel_parse_model};
  emel::model::loader::event::load load_ev{fixture.model_data, parse_model};
  load_ev.file_image = fixture.file_bytes.data();
  load_ev.file_size = fixture.file_bytes.size();
  load_ev.load_weights = {&fixture, run_emel_load_weights};
  load_ev.map_layers = {nullptr, run_emel_map_layers};
  load_ev.validate_structure = {nullptr, run_emel_validate_structure};
  load_ev.validate_architecture_impl = {nullptr, run_emel_validate_architecture};
  load_ev.on_done = {&fixture, on_load_done};
  load_ev.on_error = {&fixture, on_load_error};
  if (!fixture.model_loader.process_event(load_ev) || !fixture.load.done || fixture.load.error) {
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

llama_model_ptr load_reference_model(const std::string & model_path);
llama_context_ptr make_reference_context(
    llama_model * model,
    ggml_backend_sched_eval_callback eval_callback = nullptr,
    void * eval_user_data = nullptr);
bool run_reference_generate_preloaded(const reference_fixture & fixture,
                                      const generation_case_spec & spec,
                                      llama_context * ctx,
                                      const std::vector<llama_token> & prompt_tokens,
                                      generation_seam_audit & seam,
                                      generation_result & result_out);

bool prepare_reference_fixture(reference_fixture & fixture, const std::string & model_path) {
  fixture.model = load_reference_model(model_path);
  if (fixture.model == nullptr) {
    return false;
  }

  fixture.vocab = llama_model_get_vocab(fixture.model.get());
  if (fixture.vocab == nullptr) {
    return false;
  }

  fixture.formatter =
      emel::tools::generation_formatter_contract::resolve_reference_formatter_info(
          fixture.model.get());
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

llama_model_ptr load_reference_model(const std::string & model_path) {
  llama_model_params params = llama_model_default_params();
  // Force the reference path onto CPU so the compare stays aligned with EMEL's CPU-only backend.
  params.n_gpu_layers = 0;
  params.check_tensors = false;
  return llama_model_ptr{llama_model_load_from_file(model_path.c_str(), params), llama_model_free};
}

bool reference_graph_contains_flash_attn_op(llama_context & ctx) {
  const auto & cparams = ctx.get_cparams();
  if (!cparams.flash_attn || cparams.auto_fa) {
    return false;
  }

  llama_memory_context_ptr mctx;
  if (llama_memory_t memory = ctx.get_memory()) {
    mctx = memory->init_full();
  }

  ggml_cgraph * graph = ctx.graph_reserve(1u, 1u, 1u, mctx.get(), true);
  if (graph == nullptr) {
    return false;
  }

  for (int32_t idx = 0; idx < ggml_graph_n_nodes(graph); ++idx) {
    ggml_tensor * node = ggml_graph_node(graph, idx);
    if (node != nullptr && node->op == GGML_OP_FLASH_ATTN_EXT) {
      return true;
    }
  }
  return false;
}

llama_context_ptr init_reference_context(llama_model * model,
                                         const llama_context_params & params) {
  return llama_context_ptr{model != nullptr ? llama_init_from_model(model, params) : nullptr,
                           llama_free};
}

llama_context_ptr make_reference_context(
    llama_model * model,
    ggml_backend_sched_eval_callback eval_callback,
    void * eval_user_data) {
  llama_context_params params = llama_context_default_params();
  params.flash_attn_type = k_reference_flash_attn_type;
  params.n_ctx = 0;
  params.n_batch = 512;
  params.n_ubatch = 512;
  params.n_seq_max = 1;
  params.n_threads = 1;
  params.n_threads_batch = 1;
  params.embeddings = false;
  params.cb_eval = eval_callback;
  params.cb_eval_user_data = eval_user_data;
  llama_context_ptr probe = init_reference_context(model, params);
  if (probe != nullptr && !reference_graph_contains_flash_attn_op(*probe)) {
    fail_bench_setup("make_reference_context", "reference graph missing flash attention op");
  }
  return probe;
}

void prepare_emel_session(const emel_fixture & fixture, emel_session & session) {
  session.model_data = fixture.model_data;
  session.formatter_binding = fixture.formatter_binding;
  session.generator = std::make_unique<emel::generator::sm>(
      session.model_data,
      session.conditioner,
      session.formatter_binding.formatter_ctx,
      session.formatter_binding.format_prompt);
}

bool initialize_emel_session(emel_session & session, const generation_case_spec & spec) {
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
  const int32_t block_capacity = std::max<int32_t>(8, prompt_capacity + decode_capacity);

  reset_initialize_capture(session);
  emel::error::type error_out = emel::error::cast(emel::generator::error::none);
  emel::generator::event::initialize request{
      &session.tokenizer,
      tokenizer_bind_dispatch,
      tokenizer_tokenize_dispatch,
      std::span<emel::logits::sampler::fn>{},
  };
  request.preprocessor_variant = generation_preprocessor_variant(session.model_data);
  request.encoder_variant = generation_encoder_variant(session.model_data);
  request.add_special = false;
  request.parse_special = false;
  request.selection_mode = emel::generator::selection_mode::preselected_argmax;
  request.max_prompt_tokens = prompt_capacity;
  request.max_generated_tokens = decode_capacity;
  request.max_blocks = block_capacity;
  request.block_tokens = 16;
  request.strip_leading_space = false;
  request.error_out = &error_out;
  request.on_done = {&session, on_initialize_done};
  request.on_error = {&session, on_initialize_error};

  const bool accepted = session.generator->process_event(request);
  if ((!accepted || !session.initialize.done || session.initialize.error ||
       error_out != emel::error::cast(emel::generator::error::none)) &&
      std::getenv("EMEL_DEBUG_GENERATION_BENCH") != nullptr) {
    std::fprintf(stderr,
                 "initialize_emel_session debug accepted=%d done=%d error=%d callback_err=%s "
                 "event_err=%s arch=%.*s formatter=%.*s case=%s\n",
                 accepted ? 1 : 0,
                 session.initialize.done ? 1 : 0,
                 session.initialize.error ? 1 : 0,
                 generator_error_name(session.initialize.err).data(),
                 generator_error_name(error_out).data(),
                 static_cast<int>(emel::model::architecture_name_view(session.model_data).size()),
                 emel::model::architecture_name_view(session.model_data).data(),
                 static_cast<int>(session.formatter_binding.contract.size()),
                 session.formatter_binding.contract.data(),
                 spec.name.data());
  }
  return accepted && session.initialize.done && !session.initialize.error &&
         error_out == emel::error::cast(emel::generator::error::none);
}

bool run_emel_generate(emel_session & session,
                       const generation_case_spec & spec,
                       generation_result & result_out) {
  if (session.generator == nullptr) {
    return false;
  }

  result_out = {};
  reset_generation_capture(session);
  emel::error::type error_out = emel::error::cast(emel::generator::error::none);
  std::array<emel::text::formatter::chat_message, 1> message_storage = {};
  emel::generator::event::generate request{
      emel::tools::generation_formatter_contract::single_user_messages(
          message_storage, spec.prompt),
      spec.max_tokens,
      std::span<char>{result_out.output},
      result_out.output_length,
  };
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  request.error_out = &error_out;
  request.on_done = {&session, on_generation_done};
  request.on_error = {&session, on_generation_error};
  const bool accepted = session.generator->process_event(request);
  if ((!accepted || !session.generation.done || session.generation.error ||
       error_out != emel::error::cast(emel::generator::error::none)) &&
      std::getenv("EMEL_DEBUG_GENERATION_BENCH") != nullptr) {
    std::fprintf(stderr,
                 "run_emel_generate debug accepted=%d done=%d error=%d callback_err=%s "
                 "event_err=%s arch=%.*s formatter=%.*s case=%s\n",
                 accepted ? 1 : 0,
                 session.generation.done ? 1 : 0,
                 session.generation.error ? 1 : 0,
                 generator_error_name(session.generation.err).data(),
                 generator_error_name(error_out).data(),
                 static_cast<int>(emel::model::architecture_name_view(session.model_data).size()),
                 emel::model::architecture_name_view(session.model_data).data(),
                 static_cast<int>(session.formatter_binding.contract.size()),
                 session.formatter_binding.contract.data(),
                 spec.name.data());
  }
  if (!accepted || !session.generation.done || session.generation.error ||
      error_out != emel::error::cast(emel::generator::error::none)) {
    return false;
  }

  result_out.tokens_generated = session.generation.tokens_generated;
  result_out.output_length = session.generation.output_length;
  return true;
}

llama_token select_argmax_token_from_logits(const float * logits, const int32_t vocab_size) {
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

bool tokenize_reference_prompt(const reference_fixture & fixture,
                               const generation_case_spec & spec,
                               generation_seam_audit & seam,
                               std::vector<llama_token> & tokens_out) {
  if (fixture.vocab == nullptr) {
    return false;
  }

  std::string formatted_prompt = {};
  if (!format_direct_reference_prompt(seam, fixture.formatter, spec.prompt, formatted_prompt)) {
    return false;
  }

  int32_t token_capacity =
      std::max<int32_t>(8, static_cast<int32_t>(formatted_prompt.size()) + 8);
  tokens_out.resize(static_cast<size_t>(token_capacity));
  int32_t token_count = tokenize_direct_reference_prompt(
      seam,
      fixture.vocab,
      formatted_prompt,
      tokens_out.data(),
      token_capacity);
  if (token_count < 0) {
    token_capacity = -token_count;
    tokens_out.resize(static_cast<size_t>(token_capacity));
    token_count = tokenize_direct_reference_prompt(seam,
                                                   fixture.vocab,
                                                   formatted_prompt,
                                                   tokens_out.data(),
                                                   token_capacity);
  }
  if (token_count <= 0) {
    return false;
  }

  tokens_out.resize(static_cast<size_t>(token_count));
  return true;
}

bool append_reference_piece(const reference_fixture & fixture,
                            generation_seam_audit & seam,
                            const llama_token token,
                            generation_result & result_out) {
  if (fixture.vocab == nullptr || result_out.output_length >= result_out.output.size()) {
    return false;
  }

  if (reference_vocab_is_control(seam, fixture.vocab, token) ||
      reference_vocab_is_eog(seam, fixture.vocab, token)) {
    return true;
  }

  const char * piece = reference_vocab_text(seam, fixture.vocab, token);
  if (piece == nullptr) {
    return false;
  }

  const size_t piece_len = std::strlen(piece);
  if (result_out.output_length + piece_len > result_out.output.size()) {
    return false;
  }

  if (piece_len > 0u) {
    std::memcpy(result_out.output.data() + result_out.output_length, piece, piece_len);
  }
  result_out.output_length += piece_len;
  return true;
}

bool run_reference_generate(const reference_fixture & fixture,
                            const generation_case_spec & spec,
                            generation_seam_audit & seam,
                            generation_result & result_out) {
  if (fixture.model == nullptr || fixture.context == nullptr ||
      fixture.vocab == nullptr || fixture.vocab_size <= 0) {
    return false;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(fixture, spec, seam, prompt_tokens)) {
    return false;
  }

  return run_reference_generate_preloaded(
      fixture, spec, fixture.context.get(), prompt_tokens, seam, result_out);
}

bool reset_reference_context(llama_context * ctx) {
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

bool run_reference_generate_preloaded(const reference_fixture & fixture,
                                      const generation_case_spec & spec,
                                      llama_context * ctx,
                                      const std::vector<llama_token> & prompt_tokens,
                                      generation_seam_audit & seam,
                                      generation_result & result_out) {
  if (fixture.vocab == nullptr ||
      fixture.vocab_size <= 0 ||
      ctx == nullptr ||
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
    float * logits = read_direct_reference_logits(seam, ctx);
    if (logits == nullptr) {
      return false;
    }

    const llama_token selected = select_argmax_token_from_logits(logits, fixture.vocab_size);
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
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

std::uint64_t saturating_remainder(const std::uint64_t total,
                                   const std::uint64_t part0,
                                   const std::uint64_t part1,
                                   const std::uint64_t part2,
                                   const std::uint64_t part3) noexcept {
  const std::uint64_t measured = part0 + part1 + part2 + part3;
  return measured >= total ? 0u : total - measured;
}

std::uint64_t prefill_probe_breakdown_total(const prefill_probe_breakdown & breakdown) noexcept {
  return breakdown.linear_ns + breakdown.attention_ns + breakdown.misc_ns;
}

template <class fn_type>
bool measure_probe_bool(std::uint64_t & bucket_ns, fn_type && fn) {
  const auto start = steady_clock::now();
  const bool ok = fn();
  bucket_ns += elapsed_ns(start, steady_clock::now());
  return ok;
}

template <class fn_type>
void measure_probe_void(std::uint64_t & bucket_ns, fn_type && fn) {
  const auto start = steady_clock::now();
  fn();
  bucket_ns += elapsed_ns(start, steady_clock::now());
}

template <class fn_type>
bool measure_subprobe_bool(std::uint64_t & total_bucket_ns,
                           std::uint64_t & sub_bucket_ns,
                           fn_type && fn) {
  const auto start = steady_clock::now();
  const bool ok = fn();
  const auto elapsed = elapsed_ns(start, steady_clock::now());
  total_bucket_ns += elapsed;
  sub_bucket_ns += elapsed;
  return ok;
}

template <class fn_type>
void measure_subprobe_void(std::uint64_t & total_bucket_ns,
                           std::uint64_t & sub_bucket_ns,
                           fn_type && fn) {
  const auto start = steady_clock::now();
  fn();
  const auto elapsed = elapsed_ns(start, steady_clock::now());
  total_bucket_ns += elapsed;
  sub_bucket_ns += elapsed;
}

bool bind_emel_prefill_probe_inputs(emel::generator::detail::native_backend & backend,
                                    const std::vector<int32_t> & prompt_tokens) {
  if (prompt_tokens.empty() ||
      prompt_tokens.size() > backend.bound_tokens.size() ||
      prompt_tokens.size() > backend.bound_positions.size()) {
    return false;
  }

  std::copy(prompt_tokens.begin(), prompt_tokens.end(), backend.bound_tokens.begin());
  for (size_t index = 0; index < prompt_tokens.size(); ++index) {
    backend.bound_positions[index] = static_cast<int32_t>(index);
  }
  backend.bound_token_count = static_cast<int32_t>(prompt_tokens.size());
  backend.bound_position_count = static_cast<int32_t>(prompt_tokens.size());
  backend.bound_ready = true;
  return true;
}

enum class reference_prefill_probe_bucket : uint8_t {
  linear = 0u,
  attention = 1u,
  misc = 2u,
};

reference_prefill_probe_bucket classify_reference_prefill_tensor(const ggml_tensor * tensor) {
  if (tensor == nullptr) {
    return reference_prefill_probe_bucket::misc;
  }

  const char * op_name = ggml_op_name(tensor->op);
  if (op_name == nullptr) {
    return reference_prefill_probe_bucket::misc;
  }
  if (std::strcmp(op_name, "MUL_MAT") == 0 || std::strcmp(op_name, "MUL_MAT_ID") == 0) {
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
  reference_prefill_probe_bucket current_bucket = reference_prefill_probe_bucket::misc;
  bool current_pending = false;
};

bool observe_reference_prefill_node(ggml_tensor * tensor, const bool ask, void * user_data) {
  auto * state = static_cast<reference_prefill_probe_state *>(user_data);
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

  const std::uint64_t elapsed = elapsed_ns(state->current_start, steady_clock::now());
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

bool tokenize_conditioned_prompt(emel_session & session,
                                 const generation_case_spec & spec,
                                 std::vector<int32_t> & tokens_out) {
  const int32_t token_capacity =
      std::max<int32_t>(1024, static_cast<int32_t>(spec.prompt.size()) * 8 + 64);
  tokens_out.assign(static_cast<size_t>(token_capacity), 0);

  int32_t token_count = 0;
  int32_t conditioned_error = 0;
  std::array<emel::text::formatter::chat_message, 1> message_storage = {};
  emel::text::conditioner::event::prepare prepare_ev{token_count, conditioned_error};
  prepare_ev.messages =
      emel::tools::generation_formatter_contract::single_user_messages(message_storage, spec.prompt);
  prepare_ev.add_generation_prompt = true;
  prepare_ev.enable_thinking = false;
  prepare_ev.token_ids_out = tokens_out.data();
  prepare_ev.token_capacity = token_capacity;
  const bool accepted = session.conditioner.process_event(prepare_ev);
  if (!accepted || conditioned_error != 0 || token_count <= 0 || token_count > token_capacity) {
    return false;
  }

  tokens_out.resize(static_cast<size_t>(token_count));
  return true;
}

const char * prefill_contract_name(
    emel::generator::prefill_compute_contract contract) noexcept;
bool inspect_emel_prefill_plan(const emel::model::data & model_data,
                               const std::vector<int32_t> & prompt_tokens,
                               int32_t & step_size_out);
bool inspect_emel_prefill_contract(const emel::model::data & model_data,
                                   int32_t prompt_token_count,
                                   emel::generator::prefill_compute_contract & contract_out);

const char * prefill_contract_name(
    const emel::generator::prefill_compute_contract contract) noexcept {
  switch (contract) {
    case emel::generator::prefill_compute_contract::none:
      return "none";
    case emel::generator::prefill_compute_contract::flash_materialized_scalar:
      return "flash_materialized_scalar";
    case emel::generator::prefill_compute_contract::flash_materialized_chunk4_packed_q8_0:
      return "flash_materialized_chunk4_packed_q8_0";
    case emel::generator::prefill_compute_contract::flash_preselected_scalar:
      return "flash_preselected_scalar";
    case emel::generator::prefill_compute_contract::flash_preselected_chunk4_packed_q8_0:
      return "flash_preselected_chunk4_packed_q8_0";
    case emel::generator::prefill_compute_contract::nonflash_materialized_scalar:
      return "nonflash_materialized_scalar";
    case emel::generator::prefill_compute_contract::nonflash_materialized_chunk4_packed_q8_0:
      return "nonflash_materialized_chunk4_packed_q8_0";
    case emel::generator::prefill_compute_contract::nonflash_preselected_scalar:
      return "nonflash_preselected_scalar";
    case emel::generator::prefill_compute_contract::nonflash_preselected_chunk4_packed_q8_0:
      return "nonflash_preselected_chunk4_packed_q8_0";
    case emel::generator::prefill_compute_contract::flash_materialized_chunk4_q8_k:
      return "flash_materialized_chunk4_q8_k";
    case emel::generator::prefill_compute_contract::flash_preselected_chunk4_q8_k:
      return "flash_preselected_chunk4_q8_k";
    case emel::generator::prefill_compute_contract::nonflash_materialized_chunk4_q8_k:
      return "nonflash_materialized_chunk4_q8_k";
    case emel::generator::prefill_compute_contract::nonflash_preselected_chunk4_q8_k:
      return "nonflash_preselected_chunk4_q8_k";
    case emel::generator::prefill_compute_contract::flash_materialized_chunk8_q8_k:
      return "flash_materialized_chunk8_q8_k";
    case emel::generator::prefill_compute_contract::flash_preselected_chunk8_q8_k:
      return "flash_preselected_chunk8_q8_k";
    case emel::generator::prefill_compute_contract::nonflash_materialized_chunk8_q8_k:
      return "nonflash_materialized_chunk8_q8_k";
    case emel::generator::prefill_compute_contract::nonflash_preselected_chunk8_q8_k:
      return "nonflash_preselected_chunk8_q8_k";
  }
  return "unknown";
}

struct bench_plan_capture {
  emel::error::type err = emel::error::cast(emel::batch::planner::error::none);
  int32_t step_size = 0;
  int32_t step_count = 0;
  int32_t total_outputs = 0;
  bool done = false;
  bool error = false;
};

void capture_bench_plan_done(bench_plan_capture & capture,
                             const emel::batch::planner::events::plan_done & ev) noexcept {
  capture.done = true;
  capture.error = false;
  capture.step_size = ev.step_count > 0 && ev.step_sizes != nullptr ? ev.step_sizes[0] : 0;
  capture.step_count = ev.step_count;
  capture.total_outputs = ev.total_outputs;
}

void capture_bench_plan_error(bench_plan_capture & capture,
                              const emel::batch::planner::events::plan_error & ev) noexcept {
  capture.done = false;
  capture.error = true;
  capture.err = ev.err;
}

bool resolve_emel_prefill_plan_step(const emel::model::data & model_data,
                                    const int32_t prompt_token_count,
                                    int32_t & step_size_out) {
  emel::generator::action::context generator_context{};
  generator_context.model = &model_data;
  generator_context.state.selection_mode =
      emel::generator::selection_mode::preselected_argmax;

  const auto prepare_err = emel::generator::detail::prepare(
      generator_context.compute.backend, model_data);
  if (prepare_err !=
      emel::error::cast(emel::model::loader::error::none)) {
    return false;
  }

  std::array<char, 1> output = {};
  size_t output_length = 0;
  const std::span<const emel::text::formatter::chat_message> messages = {};
  emel::generator::event::generate request{
    messages,
    1,
    std::span<char>{output.data(), output.size()},
    output_length,
  };
  emel::generator::event::generate_ctx generate_context{};
  generate_context.prompt_token_count = prompt_token_count;
  const emel::generator::event::generate_run generate_run{request, generate_context};

  const bool uses_chunk8 =
      emel::generator::guard::planning_uses_chunk8_prefill{}(generate_run, generator_context);
  const bool uses_chunk4 =
      emel::generator::guard::planning_uses_chunk4_prefill{}(generate_run, generator_context);
  if (uses_chunk8) {
    step_size_out = emel::generator::detail::k_prefill_q8_chunk8_rows;
    return true;
  }
  step_size_out = uses_chunk4 ? emel::generator::detail::k_prefill_q8_chunk_rows : 1;
  return true;
}

bool inspect_emel_prefill_plan(const emel::model::data & model_data,
                               const std::vector<int32_t> & prompt_tokens,
                               int32_t & step_size_out) {
  int32_t resolved_step_size = 0;
  if (!resolve_emel_prefill_plan_step(
          model_data,
          static_cast<int32_t>(prompt_tokens.size()),
          resolved_step_size)) {
    return false;
  }

  bench_plan_capture capture = {};
  emel::batch::planner::sm planner = {};
  const auto on_done =
      emel::callback<void(const emel::batch::planner::events::plan_done &)>::from<
          bench_plan_capture,
          capture_bench_plan_done>(&capture);
  const auto on_error =
      emel::callback<void(const emel::batch::planner::events::plan_error &)>::from<
          bench_plan_capture,
          capture_bench_plan_error>(&capture);
  emel::batch::planner::event::plan_request request{
    .token_ids = prompt_tokens.data(),
    .n_tokens = static_cast<int32_t>(prompt_tokens.size()),
    .n_steps = resolved_step_size,
    .mode = emel::batch::planner::event::plan_mode::simple,
    .seq_masks = nullptr,
    .seq_masks_count = 0,
    .seq_primary_ids = nullptr,
    .seq_primary_ids_count = 0,
    .equal_sequential = true,
    .seq_mask_words = emel::generator::action::k_sequence_mask_words,
    .output_mask = nullptr,
    .output_mask_count = 0,
    .output_all = false,
    .on_done = on_done,
    .on_error = on_error,
  };
  if (!planner.process_event(request) || capture.error || !capture.done) {
    return false;
  }
  step_size_out = capture.step_size;
  return step_size_out > 0;
}

bool inspect_emel_prefill_contract(const emel::model::data & model_data,
                                   const int32_t prompt_token_count,
                                   emel::generator::prefill_compute_contract & contract_out) {
  emel::generator::action::context generator_context{};
  generator_context.model = &model_data;
  generator_context.state.selection_mode =
      emel::generator::selection_mode::preselected_argmax;
  if (emel::generator::detail::prepare(generator_context.compute.backend, model_data) !=
      emel::error::cast(emel::model::loader::error::none)) {
    return false;
  }

  emel::generator::prefill::action::context prefill_context{generator_context};
  std::array<emel::text::formatter::chat_message, 1> message_storage = {};
  std::array<char, 1> output_storage = {};
  std::size_t output_length = 0u;
  emel::generator::event::generate request{
      emel::tools::generation_formatter_contract::single_user_messages(message_storage, "probe"),
      1,
      std::span<char>{output_storage.data(), output_storage.size()},
      output_length,
  };
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  emel::generator::event::generate_ctx generate_context{};
  generate_context.prompt_token_count = prompt_token_count;
  emel::generator::prefill::event::run prefill_run{request, generate_context};

  if (emel::generator::prefill::guard::flash_runtime_supported{}(
          prefill_run, prefill_context)) {
    if (emel::generator::prefill::guard::uses_materialized_logits_with_chunk8_q8_k{}(
            prefill_run, prefill_context)) {
      contract_out =
          emel::generator::prefill_compute_contract::flash_materialized_chunk8_q8_k;
      return true;
    }
    if (emel::generator::prefill::guard::uses_materialized_logits_with_chunk4_packed_q8_0{}(
            prefill_run, prefill_context)) {
      contract_out =
          emel::generator::prefill_compute_contract::flash_materialized_chunk4_packed_q8_0;
      return true;
    }
    if (emel::generator::prefill::guard::uses_materialized_logits_with_chunk4_q8_k{}(
            prefill_run, prefill_context)) {
      contract_out =
          emel::generator::prefill_compute_contract::flash_materialized_chunk4_q8_k;
      return true;
    }
    if (emel::generator::prefill::guard::uses_materialized_logits_with_scalar{}(
            prefill_run, prefill_context)) {
      contract_out = emel::generator::prefill_compute_contract::flash_materialized_scalar;
      return true;
    }
    if (emel::generator::prefill::guard::uses_preselected_argmax_with_chunk4_packed_q8_0{}(
            prefill_run, prefill_context)) {
      contract_out =
          emel::generator::prefill_compute_contract::flash_preselected_chunk4_packed_q8_0;
      return true;
    }
    if (emel::generator::prefill::guard::uses_preselected_argmax_with_chunk8_q8_k{}(
            prefill_run, prefill_context)) {
      contract_out = emel::generator::prefill_compute_contract::flash_preselected_chunk8_q8_k;
      return true;
    }
    if (emel::generator::prefill::guard::uses_preselected_argmax_with_chunk4_q8_k{}(
            prefill_run, prefill_context)) {
      contract_out = emel::generator::prefill_compute_contract::flash_preselected_chunk4_q8_k;
      return true;
    }
    if (emel::generator::prefill::guard::uses_preselected_argmax_with_scalar{}(
            prefill_run, prefill_context)) {
      contract_out = emel::generator::prefill_compute_contract::flash_preselected_scalar;
      return true;
    }
    return false;
  }

  if (emel::generator::prefill::guard::uses_materialized_logits_with_chunk8_q8_k{}(
          prefill_run, prefill_context)) {
    contract_out = emel::generator::prefill_compute_contract::nonflash_materialized_chunk8_q8_k;
    return true;
  }
  if (emel::generator::prefill::guard::uses_materialized_logits_with_chunk4_packed_q8_0{}(
          prefill_run, prefill_context)) {
    contract_out =
        emel::generator::prefill_compute_contract::nonflash_materialized_chunk4_packed_q8_0;
    return true;
  }
  if (emel::generator::prefill::guard::uses_materialized_logits_with_chunk4_q8_k{}(
          prefill_run, prefill_context)) {
    contract_out = emel::generator::prefill_compute_contract::nonflash_materialized_chunk4_q8_k;
    return true;
  }
  if (emel::generator::prefill::guard::uses_materialized_logits_with_scalar{}(
          prefill_run, prefill_context)) {
    contract_out = emel::generator::prefill_compute_contract::nonflash_materialized_scalar;
    return true;
  }
  if (emel::generator::prefill::guard::uses_preselected_argmax_with_chunk4_packed_q8_0{}(
          prefill_run, prefill_context)) {
    contract_out =
        emel::generator::prefill_compute_contract::nonflash_preselected_chunk4_packed_q8_0;
    return true;
  }
  if (emel::generator::prefill::guard::uses_preselected_argmax_with_chunk8_q8_k{}(
          prefill_run, prefill_context)) {
    contract_out = emel::generator::prefill_compute_contract::nonflash_preselected_chunk8_q8_k;
    return true;
  }
  if (emel::generator::prefill::guard::uses_preselected_argmax_with_chunk4_q8_k{}(
          prefill_run, prefill_context)) {
    contract_out = emel::generator::prefill_compute_contract::nonflash_preselected_chunk4_q8_k;
    return true;
  }
  if (emel::generator::prefill::guard::uses_preselected_argmax_with_scalar{}(
          prefill_run, prefill_context)) {
    contract_out = emel::generator::prefill_compute_contract::nonflash_preselected_scalar;
    return true;
  }
  return false;
}

bool initialize_emel_renderer(const emel::model::data & model_data,
                              emel::text::renderer::sm & renderer) {
  int32_t renderer_err = emel::error::cast(emel::text::renderer::error::none);
  emel::text::renderer::event::initialize initialize_renderer_ev{model_data.vocab_data};
  initialize_renderer_ev.strip_leading_space = false;
  initialize_renderer_ev.stop_sequences = nullptr;
  initialize_renderer_ev.stop_sequence_count = 0;
  initialize_renderer_ev.error_out = &renderer_err;
  return renderer.process_event(initialize_renderer_ev) &&
         renderer_err == emel::error::cast(emel::text::renderer::error::none);
}

bool append_rendered_token(emel::text::renderer::sm & renderer,
                           const int32_t token_id,
                           generation_result & result_out) {
  if (result_out.output_length > result_out.output.size()) {
    return false;
  }

  size_t appended_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t render_err = emel::error::cast(emel::text::renderer::error::none);
  emel::text::renderer::event::render render_ev = {};
  render_ev.token_id = token_id;
  render_ev.sequence_id = 0;
  render_ev.emit_special = false;
  render_ev.output = result_out.output.data() + result_out.output_length;
  render_ev.output_capacity = result_out.output.size() - result_out.output_length;
  render_ev.output_length_out = &appended_length;
  render_ev.status_out = &status;
  render_ev.error_out = &render_err;
  if (!renderer.process_event(render_ev) ||
      render_err != emel::error::cast(emel::text::renderer::error::none)) {
    return false;
  }

  result_out.output_length += appended_length;
  return true;
}

bool flush_rendered_output(emel::text::renderer::sm & renderer,
                           generation_result & result_out) {
  if (result_out.output_length > result_out.output.size()) {
    return false;
  }

  size_t flush_length = 0;
  emel::text::renderer::sequence_status status =
      emel::text::renderer::sequence_status::running;
  int32_t flush_err = emel::error::cast(emel::text::renderer::error::none);
  emel::text::renderer::event::flush flush_ev = {};
  flush_ev.sequence_id = 0;
  flush_ev.output = result_out.output.data() + result_out.output_length;
  flush_ev.output_capacity = result_out.output.size() - result_out.output_length;
  flush_ev.output_length_out = &flush_length;
  flush_ev.status_out = &status;
  flush_ev.error_out = &flush_err;
  if (!renderer.process_event(flush_ev) ||
      flush_err != emel::error::cast(emel::text::renderer::error::none)) {
    return false;
  }

  result_out.output_length += flush_length;
  return true;
}

template <emel::generator::attention_mode mode>
bool run_emel_runtime_layer_probe_scalar(emel::generator::detail::native_backend & backend,
                                         const int32_t layer_index,
                                         const int32_t position,
                                         prefill_probe_breakdown & breakdown) {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  auto q = std::span<float>(backend.q.data(), static_cast<size_t>(block.attention_q_dim));
  auto k = std::span<float>(backend.k.data(), static_cast<size_t>(block.attention_kv_dim));
  auto v = std::span<float>(backend.v.data(), static_cast<size_t>(block.attention_kv_dim));
  auto attn_ctx =
      std::span<float>(backend.attn_ctx.data(), static_cast<size_t>(block.attention_q_dim));
  if (!measure_probe_bool(
          breakdown.misc_ns,
          [&]() {
            return emel::generator::detail::rms_norm(
                backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm);
          })) {
    return false;
  }

  if (block.uses_attention) {
    const bool qkv_shared_packed_q8_0 =
        emel::generator::detail::packed_q8_0_input_path_supported(backend, block.attention_q) &&
        emel::generator::detail::packed_q8_0_input_path_supported(backend, block.attention_k) &&
        emel::generator::detail::packed_q8_0_input_path_supported(backend, block.attention_v);
    const bool qkv_shared_q8 =
        emel::generator::detail::q8_input_path_supported(backend, block.attention_q) &&
        emel::generator::detail::q8_input_path_supported(backend, block.attention_k) &&
        emel::generator::detail::q8_input_path_supported(backend, block.attention_v);
    if (qkv_shared_packed_q8_0) {
      if (!measure_probe_bool(
              breakdown.linear_ns,
              [&]() {
                return emel::generator::detail::prepare_packed_q8_0_input(backend, backend.norm) &&
                    emel::generator::detail::matmul_vector_prepared_packed_q8_0_input(
                               backend, block.attention_q, block.attention_q.cols, q) &&
                    emel::generator::detail::matmul_vector_prepared_packed_q8_0_input(
                               backend, block.attention_k, block.attention_k.cols, k) &&
                    emel::generator::detail::matmul_vector_prepared_packed_q8_0_input(
                               backend, block.attention_v, block.attention_v.cols, v);
              })) {
        return false;
      }
    } else if (qkv_shared_q8) {
      const size_t block_count =
          static_cast<size_t>(backend.n_embd) /
          static_cast<size_t>(emel::generator::detail::quant::QK_K);
      if (block_count == 0u || block_count > backend.q8_input_storage.size()) {
        return false;
      }
      auto q8_input = std::span<emel::kernel::detail::quant::block_q8_k>(
          backend.q8_input_storage.data(), block_count);
      if (!measure_probe_bool(
              breakdown.linear_ns,
              [&]() {
                return emel::generator::detail::quantize_vector_q8_k(backend.norm, q8_input) &&
                    emel::generator::detail::matmul_vector_q8_input(
                               backend, block.attention_q, q8_input, block.attention_q.cols, q) &&
                    emel::generator::detail::matmul_vector_q8_input(
                               backend, block.attention_k, q8_input, block.attention_k.cols, k) &&
                    emel::generator::detail::matmul_vector_q8_input(
                               backend, block.attention_v, q8_input, block.attention_v.cols, v);
              })) {
        return false;
      }
    } else if (!measure_probe_bool(
                   breakdown.linear_ns,
                   [&]() {
                     return emel::generator::detail::matmul_vector(
                                backend, block.attention_q, backend.norm, q) &&
                         emel::generator::detail::matmul_vector(
                                backend, block.attention_k, backend.norm, k) &&
                         emel::generator::detail::matmul_vector(
                                backend, block.attention_v, backend.norm, v);
                   })) {
      return false;
    }

    if (emel::generator::detail::requires_attention_qk_norm(backend, block) &&
        !measure_probe_bool(
            breakdown.misc_ns,
            [&]() { return emel::generator::detail::apply_attention_qk_norm(backend, block); })) {
      return false;
    }

    measure_probe_void(
        breakdown.misc_ns,
        [&]() {
          emel::generator::detail::apply_rope(
              q,
              backend.n_head,
              block.attention_head_dim,
              block.attention_rope_dim,
              position,
              block.attention_rope_freq_base);
          emel::generator::detail::apply_rope(
              k,
              backend.n_head_kv,
              block.attention_head_dim_kv,
              block.attention_rope_dim,
              position,
              block.attention_rope_freq_base);
        });

    if (!measure_probe_bool(
            breakdown.misc_ns,
            [&]() {
              return emel::generator::detail::store_attention_kv_cache(
                  backend,
                  block,
                  layer_index,
                  position,
                  std::span<const float>(k.data(), k.size()),
                  std::span<const float>(v.data(), v.size()));
            }) ||
        !measure_probe_bool(
            breakdown.attention_ns,
            [&]() {
              return emel::generator::detail::run_attention<mode>(
                  backend, block, layer_index, position);
            }) ||
        !measure_probe_bool(
            breakdown.linear_ns,
            [&]() {
              return emel::generator::detail::matmul_vector(
                  backend, block.attention_output, attn_ctx, backend.projected);
            })) {
      return false;
    }

    measure_probe_void(
        breakdown.misc_ns,
        [&]() {
          for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
            backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
          }
        });
  } else if (!measure_probe_bool(
                 breakdown.misc_ns,
                 [&]() {
                   return emel::generator::detail::run_shortconv_block(
                       backend, block, layer_index);
                 })) {
    return false;
  }

  if (!measure_probe_bool(
          breakdown.misc_ns,
          [&]() {
            return emel::generator::detail::rms_norm(
                backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm);
          })) {
    return false;
  }

  const int32_t ffn_dim = block.feed_forward_gate.rows;
  auto gate = std::span<float>(backend.gate.data(), static_cast<size_t>(ffn_dim));
  auto up = std::span<float>(backend.up.data(), static_cast<size_t>(ffn_dim));
  auto ffn_hidden = std::span<float>(backend.ffn_hidden.data(), static_cast<size_t>(ffn_dim));
  const bool gate_up_shared_packed_q8_0 =
      emel::generator::detail::packed_q8_0_input_path_supported(backend, block.feed_forward_gate) &&
      emel::generator::detail::packed_q8_0_input_path_supported(backend, block.feed_forward_up);
  const bool gate_up_shared_q8 =
      emel::generator::detail::q8_input_path_supported(backend, block.feed_forward_gate) &&
      emel::generator::detail::q8_input_path_supported(backend, block.feed_forward_up);
  if (gate_up_shared_packed_q8_0) {
    if (!measure_probe_bool(
            breakdown.linear_ns,
            [&]() {
              return emel::generator::detail::prepare_packed_q8_0_input(backend, backend.norm) &&
                  emel::generator::detail::matmul_vector_prepared_packed_q8_0_input(
                             backend,
                             block.feed_forward_gate,
                             block.feed_forward_gate.cols,
                             gate) &&
                  emel::generator::detail::matmul_vector_prepared_packed_q8_0_input(
                             backend, block.feed_forward_up, block.feed_forward_up.cols, up);
            })) {
      return false;
    }
  } else if (gate_up_shared_q8) {
    const size_t block_count =
        static_cast<size_t>(backend.n_embd) /
        static_cast<size_t>(emel::generator::detail::quant::QK_K);
    if (block_count == 0u || block_count > backend.q8_input_storage.size()) {
      return false;
    }
    auto q8_input = std::span<emel::kernel::detail::quant::block_q8_k>(
        backend.q8_input_storage.data(), block_count);
    if (!measure_probe_bool(
            breakdown.linear_ns,
            [&]() {
              return emel::generator::detail::quantize_vector_q8_k(backend.norm, q8_input) &&
                  emel::generator::detail::matmul_vector_q8_input(
                             backend,
                             block.feed_forward_gate,
                             q8_input,
                             block.feed_forward_gate.cols,
                             gate) &&
                  emel::generator::detail::matmul_vector_q8_input(
                             backend,
                             block.feed_forward_up,
                             q8_input,
                             block.feed_forward_up.cols,
                             up);
            })) {
      return false;
    }
  } else if (!measure_probe_bool(
                 breakdown.linear_ns,
                 [&]() {
                   return emel::generator::detail::matmul_vector(
                              backend, block.feed_forward_gate, backend.norm, gate) &&
                       emel::generator::detail::matmul_vector(
                              backend, block.feed_forward_up, backend.norm, up);
                 })) {
    return false;
  }

  measure_probe_void(
      breakdown.misc_ns,
      [&]() {
        for (size_t idx = 0; idx < gate.size(); ++idx) {
          ffn_hidden[idx] = emel::generator::detail::silu(gate[idx]) * up[idx];
        }
      });

  if (!measure_probe_bool(
          breakdown.linear_ns,
          [&]() {
            return emel::generator::detail::matmul_vector(
                backend, block.feed_forward_down, ffn_hidden, backend.projected);
          })) {
    return false;
  }

  measure_probe_void(
      breakdown.misc_ns,
      [&]() {
        for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
          backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
        }
      });
  return true;
}

template <emel::generator::attention_mode mode, emel::generator::detail::chunk4_rhs_route route>
bool run_emel_runtime_layer_probe_chunk4(emel::generator::detail::native_backend & backend,
                                         const int32_t layer_index,
                                         const size_t token_base,
                                         prefill_probe_breakdown & breakdown) {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  const int32_t q_dim = block.attention_q_dim;
  const int32_t kv_dim = block.attention_kv_dim;
  const int32_t ffn_dim = block.feed_forward_gate.rows;

  if (!measure_subprobe_bool(
          breakdown.misc_ns,
          breakdown.misc_attention_norm_ns,
          [&]() {
            return emel::generator::detail::rms_norm_chunk4(
                backend.hidden_chunk4,
                backend.n_embd,
                block.attention_norm,
                backend.rms_epsilon,
                backend.norm_chunk4);
          })) {
    return false;
  }
  if (block.uses_attention) {
    if (!measure_probe_bool(
            breakdown.linear_ns,
            [&]() {
              return emel::generator::detail::prepare_chunk4_rhs<route>(
                         backend, backend.norm_chunk4, backend.n_embd) &&
                  emel::generator::detail::matmul_chunk4_prepared<route>(
                             backend, block.attention_q, backend.n_embd, backend.q_chunk4) &&
                  emel::generator::detail::matmul_chunk4_prepared<route>(
                             backend, block.attention_k, backend.n_embd, backend.k_chunk4) &&
                  emel::generator::detail::matmul_chunk4_prepared<route>(
                             backend, block.attention_v, backend.n_embd, backend.v_chunk4);
            })) {
      return false;
    }

    const bool qk_norm_runtime = emel::generator::detail::requires_attention_qk_norm(backend, block);
    for (int32_t row = 0; row < emel::generator::detail::k_prefill_q8_chunk_rows; ++row) {
      const int32_t position = backend.bound_positions[token_base + static_cast<size_t>(row)];
      auto q_row = emel::generator::detail::chunk4_row_span<float>(
          std::span<float>(backend.q_chunk4), row, q_dim);
      auto k_row = emel::generator::detail::chunk4_row_span<float>(
          std::span<float>(backend.k_chunk4), row, kv_dim);
      const auto v_row = emel::generator::detail::chunk4_row_span<const float>(
          std::span<const float>(backend.v_chunk4), row, kv_dim);

      if (qk_norm_runtime &&
          !measure_subprobe_bool(
              breakdown.misc_ns,
              breakdown.misc_qk_norm_ns,
              [&]() {
                return emel::generator::detail::apply_headwise_rms_norm(
                           q_row,
                           block.attention_q_norm,
                           backend.n_head,
                           block.attention_head_dim,
                           backend.rms_epsilon) &&
                    emel::generator::detail::apply_headwise_rms_norm(
                           k_row,
                           block.attention_k_norm,
                           backend.n_head_kv,
                           block.attention_head_dim_kv,
                           backend.rms_epsilon);
              })) {
        return false;
      }
      measure_subprobe_void(
          breakdown.misc_ns,
          breakdown.misc_rope_ns,
          [&]() {
            emel::generator::detail::apply_rope(
                q_row,
                backend.n_head,
                block.attention_head_dim,
                block.attention_rope_dim,
                position,
                block.attention_rope_freq_base);
            emel::generator::detail::apply_rope(
                k_row,
                backend.n_head_kv,
                block.attention_head_dim_kv,
                block.attention_rope_dim,
                position,
                block.attention_rope_freq_base);
          });

      if (!measure_subprobe_bool(
              breakdown.misc_ns,
              breakdown.misc_kv_store_ns,
              [&]() {
                return emel::generator::detail::store_attention_kv_cache(
                    backend, block, layer_index, position, k_row, v_row);
              }) ||
          !measure_probe_bool(
              breakdown.attention_ns,
              [&]() {
                return emel::generator::detail::run_attention_for_q_vector<mode>(
                    backend, block, layer_index, position, q_row);
              })) {
        return false;
      }

      measure_subprobe_void(
          breakdown.misc_ns,
          breakdown.misc_ctx_copy_ns,
          [&]() {
            std::copy(
                backend.attn_ctx.begin(),
                backend.attn_ctx.begin() + q_dim,
                emel::generator::detail::chunk4_row_span<float>(
                    std::span<float>(backend.attn_ctx_chunk4), row, q_dim)
                    .begin());
            backend.kv_cache_tokens = position + 1;
          });
    }

    if (!measure_probe_bool(
            breakdown.linear_ns,
            [&]() {
              return emel::generator::detail::prepare_chunk4_rhs<route>(
                         backend, backend.attn_ctx_chunk4, q_dim) &&
                  emel::generator::detail::matmul_chunk4_prepared<route>(
                             backend, block.attention_output, q_dim, backend.projected_chunk4) &&
                  emel::generator::detail::add_chunk4_rows_in_place(
                             backend.hidden_chunk4, backend.projected_chunk4, backend.n_embd);
            })) {
      return false;
    }
  } else {
    if (backend.shortconv_kernel_size <= 0 ||
        backend.shortconv_state_size <= 0 ||
        block.shortconv_in_proj.tensor == nullptr ||
        block.shortconv_out_proj.tensor == nullptr ||
        static_cast<size_t>(block.shortconv_in_proj.rows) !=
            static_cast<size_t>(3 * backend.n_embd) ||
        block.shortconv_in_proj.cols != backend.n_embd ||
        static_cast<size_t>(block.shortconv_out_proj.rows) !=
            static_cast<size_t>(backend.n_embd) ||
        block.shortconv_out_proj.cols != backend.n_embd ||
        block.shortconv_conv.size() !=
            static_cast<size_t>(backend.shortconv_kernel_size) *
                static_cast<size_t>(backend.n_embd) ||
        backend.shortconv_bcx_chunk4.size() !=
            static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk_rows) *
                static_cast<size_t>(3 * backend.n_embd) ||
        backend.shortconv_bx.size() != static_cast<size_t>(backend.n_embd) ||
        backend.shortconv_conv_out_chunk4.size() != backend.hidden_chunk4.size()) {
      return false;
    }

    if (!measure_subprobe_bool(
            breakdown.linear_ns,
            breakdown.shortconv_in_proj_prepare_ns,
            [&]() {
              return emel::generator::detail::prepare_chunk4_rhs<route>(
                  backend, backend.norm_chunk4, backend.n_embd);
            }) ||
        !measure_subprobe_bool(
            breakdown.linear_ns,
            breakdown.shortconv_in_proj_ns,
            [&]() {
              return emel::generator::detail::matmul_chunk4_prepared<route>(
                  backend,
                  block.shortconv_in_proj,
                  backend.n_embd,
                  backend.shortconv_bcx_chunk4);
            })) {
      return false;
    }

    const size_t layer_offset = emel::generator::detail::shortconv_state_layer_offset(
        backend, layer_index);
    float * state = backend.recurrent_shortconv_cache.data() + layer_offset;
    for (int32_t row = 0; row < emel::generator::detail::k_prefill_q8_chunk_rows; ++row) {
      const auto bcx_row = emel::generator::detail::chunk4_row_span<const float>(
          std::span<const float>(backend.shortconv_bcx_chunk4), row, 3 * backend.n_embd);
      auto conv_out_row = emel::generator::detail::chunk4_row_span<float>(
          std::span<float>(backend.shortconv_conv_out_chunk4), row, backend.n_embd);
      auto b = bcx_row.subspan(0u, static_cast<size_t>(backend.n_embd));
      auto c = bcx_row.subspan(
          static_cast<size_t>(backend.n_embd), static_cast<size_t>(backend.n_embd));
      auto x = bcx_row.subspan(
          static_cast<size_t>(2 * backend.n_embd), static_cast<size_t>(backend.n_embd));

      measure_subprobe_void(
          breakdown.misc_ns,
          breakdown.shortconv_conv_ns,
          [&]() {
            for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
              const size_t dim = static_cast<size_t>(idx);
              const float bx = b[dim] * x[dim];
              backend.shortconv_bx[dim] = bx;

              const float * kernel =
                  block.shortconv_conv.data() +
                  (dim * static_cast<size_t>(backend.shortconv_kernel_size));
              float conv_sum = bx * kernel[static_cast<size_t>(backend.shortconv_state_size)];
              for (int32_t tap = 0; tap < backend.shortconv_state_size; ++tap) {
                conv_sum +=
                    state[static_cast<size_t>(tap) * static_cast<size_t>(backend.n_embd) + dim] *
                    kernel[static_cast<size_t>(tap)];
              }

              conv_out_row[dim] = c[dim] * conv_sum;
            }
          });

      measure_subprobe_void(
          breakdown.misc_ns,
          breakdown.shortconv_state_shift_ns,
          [&]() {
            if (backend.shortconv_state_size > 1) {
              const size_t move_count =
                  static_cast<size_t>(backend.shortconv_state_size - 1) *
                  static_cast<size_t>(backend.n_embd);
              std::memmove(
                  state,
                  state + static_cast<size_t>(backend.n_embd),
                  move_count * sizeof(float));
            }
            std::memcpy(
                state + static_cast<size_t>(backend.shortconv_state_size - 1) *
                            static_cast<size_t>(backend.n_embd),
                backend.shortconv_bx.data(),
                static_cast<size_t>(backend.n_embd) * sizeof(float));
          });
    }

    if (!measure_subprobe_bool(
            breakdown.linear_ns,
            breakdown.shortconv_out_proj_prepare_ns,
            [&]() {
              return emel::generator::detail::prepare_chunk4_rhs<route>(
                  backend, backend.shortconv_conv_out_chunk4, backend.n_embd);
            }) ||
        !measure_subprobe_bool(
            breakdown.linear_ns,
            breakdown.shortconv_out_proj_ns,
            [&]() {
              return emel::generator::detail::matmul_chunk4_prepared<route>(
                         backend,
                         block.shortconv_out_proj,
                         backend.n_embd,
                         backend.projected_chunk4) &&
                  emel::generator::detail::add_chunk4_rows_in_place(
                      backend.hidden_chunk4, backend.projected_chunk4, backend.n_embd);
            })) {
      return false;
    }

    breakdown.misc_shortconv_ns =
        breakdown.shortconv_conv_ns + breakdown.shortconv_state_shift_ns;
  }

  if (!measure_subprobe_bool(
          breakdown.misc_ns,
          breakdown.misc_ffn_norm_ns,
          [&]() {
            return emel::generator::detail::rms_norm_chunk4(
                backend.hidden_chunk4,
                backend.n_embd,
                block.feed_forward_norm,
                backend.rms_epsilon,
                backend.norm_chunk4);
          })) {
    return false;
  }

  if (!measure_probe_bool(
          breakdown.linear_ns,
          [&]() {
            auto gate_chunk =
                std::span<float>(backend.gate_chunk4.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk_rows) *
                                     static_cast<size_t>(ffn_dim));
            auto up_chunk =
                std::span<float>(backend.up_chunk4.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk_rows) *
                                     static_cast<size_t>(ffn_dim));
            return emel::generator::detail::prepare_chunk4_rhs<route>(
                       backend, backend.norm_chunk4, backend.n_embd) &&
                emel::generator::detail::matmul_chunk4_prepared<route>(
                           backend, block.feed_forward_gate, backend.n_embd, gate_chunk) &&
                emel::generator::detail::matmul_chunk4_prepared<route>(
                           backend, block.feed_forward_up, backend.n_embd, up_chunk);
          })) {
    return false;
  }

  if (!measure_subprobe_bool(
          breakdown.misc_ns,
          breakdown.misc_silu_ns,
          [&]() {
            auto gate_chunk =
                std::span<float>(backend.gate_chunk4.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk_rows) *
                                     static_cast<size_t>(ffn_dim));
            auto up_chunk =
                std::span<float>(backend.up_chunk4.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk_rows) *
                                     static_cast<size_t>(ffn_dim));
            auto ffn_hidden_chunk =
                std::span<float>(backend.ffn_hidden_chunk4.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk_rows) *
                                     static_cast<size_t>(ffn_dim));
            return emel::generator::detail::apply_silu_mul_chunk4(
                gate_chunk, up_chunk, ffn_dim, ffn_hidden_chunk);
          })) {
    return false;
  }

  if (!measure_probe_bool(
          breakdown.linear_ns,
          [&]() {
            auto ffn_hidden_chunk =
                std::span<float>(backend.ffn_hidden_chunk4.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk_rows) *
                                     static_cast<size_t>(ffn_dim));
            return emel::generator::detail::prepare_chunk4_rhs<route>(
                       backend, ffn_hidden_chunk, ffn_dim) &&
                emel::generator::detail::matmul_chunk4_prepared<route>(
                           backend, block.feed_forward_down, ffn_dim, backend.projected_chunk4) &&
                emel::generator::detail::add_chunk4_rows_in_place(
                           backend.hidden_chunk4, backend.projected_chunk4, backend.n_embd);
          })) {
    return false;
  }

  return true;
}

template <emel::generator::attention_mode mode>
bool run_emel_runtime_layer_probe_chunk8(emel::generator::detail::native_backend & backend,
                                         const int32_t layer_index,
                                         const size_t token_base,
                                         prefill_probe_breakdown & breakdown) {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  const int32_t q_dim = block.attention_q_dim;
  const int32_t kv_dim = block.attention_kv_dim;
  const int32_t ffn_dim = block.feed_forward_gate.rows;

  if (!measure_subprobe_bool(
          breakdown.misc_ns,
          breakdown.misc_attention_norm_ns,
          [&]() {
            return emel::generator::detail::rms_norm_chunk8(
                backend.hidden_chunk8,
                backend.n_embd,
                block.attention_norm,
                backend.rms_epsilon,
                backend.norm_chunk8);
          })) {
    return false;
  }
  if (block.uses_attention) {
    if (!measure_probe_bool(
            breakdown.linear_ns,
            [&]() {
              return emel::generator::detail::prepare_q8_chunk8_input(
                         backend, backend.norm_chunk8, backend.n_embd) &&
                  emel::generator::detail::matmul_chunk8_q8_input(
                             backend, block.attention_q, backend.n_embd, backend.q_chunk8) &&
                  emel::generator::detail::matmul_chunk8_q8_input(
                             backend, block.attention_k, backend.n_embd, backend.k_chunk8) &&
                  emel::generator::detail::matmul_chunk8_q8_input(
                             backend, block.attention_v, backend.n_embd, backend.v_chunk8);
            })) {
      return false;
    }

    const bool qk_norm_runtime = emel::generator::detail::requires_attention_qk_norm(backend, block);
    for (int32_t row = 0; row < emel::generator::detail::k_prefill_q8_chunk8_rows; ++row) {
      const int32_t position = backend.bound_positions[token_base + static_cast<size_t>(row)];
      auto q_row = emel::generator::detail::chunk8_row_span<float>(
          std::span<float>(backend.q_chunk8), row, q_dim);
      auto k_row = emel::generator::detail::chunk8_row_span<float>(
          std::span<float>(backend.k_chunk8), row, kv_dim);
      const auto v_row = emel::generator::detail::chunk8_row_span<const float>(
          std::span<const float>(backend.v_chunk8), row, kv_dim);

      if (qk_norm_runtime &&
          !measure_subprobe_bool(
              breakdown.misc_ns,
              breakdown.misc_qk_norm_ns,
              [&]() {
                return emel::generator::detail::apply_headwise_rms_norm(
                           q_row,
                           block.attention_q_norm,
                           backend.n_head,
                           block.attention_head_dim,
                           backend.rms_epsilon) &&
                    emel::generator::detail::apply_headwise_rms_norm(
                           k_row,
                           block.attention_k_norm,
                           backend.n_head_kv,
                           block.attention_head_dim_kv,
                           backend.rms_epsilon);
              })) {
        return false;
      }
      measure_subprobe_void(
          breakdown.misc_ns,
          breakdown.misc_rope_ns,
          [&]() {
            emel::generator::detail::apply_rope(
                q_row,
                backend.n_head,
                block.attention_head_dim,
                block.attention_rope_dim,
                position,
                block.attention_rope_freq_base);
            emel::generator::detail::apply_rope(
                k_row,
                backend.n_head_kv,
                block.attention_head_dim_kv,
                block.attention_rope_dim,
                position,
                block.attention_rope_freq_base);
          });

      if (!measure_subprobe_bool(
              breakdown.misc_ns,
              breakdown.misc_kv_store_ns,
              [&]() {
                return emel::generator::detail::store_attention_kv_cache(
                    backend, block, layer_index, position, k_row, v_row);
              }) ||
          !measure_probe_bool(
              breakdown.attention_ns,
              [&]() {
                return emel::generator::detail::run_attention_for_q_vector<mode>(
                    backend, block, layer_index, position, q_row);
              })) {
        return false;
      }

      measure_subprobe_void(
          breakdown.misc_ns,
          breakdown.misc_ctx_copy_ns,
          [&]() {
            std::copy(
                backend.attn_ctx.begin(),
                backend.attn_ctx.begin() + q_dim,
                emel::generator::detail::chunk8_row_span<float>(
                    std::span<float>(backend.attn_ctx_chunk8), row, q_dim)
                    .begin());
            backend.kv_cache_tokens = position + 1;
          });
    }

    if (!measure_probe_bool(
            breakdown.linear_ns,
            [&]() {
              return emel::generator::detail::prepare_q8_chunk8_input(
                         backend, backend.attn_ctx_chunk8, q_dim) &&
                  emel::generator::detail::matmul_chunk8_q8_input(
                             backend, block.attention_output, q_dim, backend.projected_chunk8) &&
                  emel::generator::detail::add_chunk8_rows_in_place(
                             backend.hidden_chunk8, backend.projected_chunk8, backend.n_embd);
            })) {
      return false;
    }
  } else {
    if (backend.shortconv_kernel_size <= 0 ||
        backend.shortconv_state_size <= 0 ||
        block.shortconv_in_proj.tensor == nullptr ||
        block.shortconv_out_proj.tensor == nullptr ||
        static_cast<size_t>(block.shortconv_in_proj.rows) !=
            static_cast<size_t>(3 * backend.n_embd) ||
        block.shortconv_in_proj.cols != backend.n_embd ||
        static_cast<size_t>(block.shortconv_out_proj.rows) !=
            static_cast<size_t>(backend.n_embd) ||
        block.shortconv_out_proj.cols != backend.n_embd ||
        block.shortconv_conv.size() !=
            static_cast<size_t>(backend.shortconv_kernel_size) *
                static_cast<size_t>(backend.n_embd) ||
        backend.shortconv_bcx_chunk8.size() !=
            static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk8_rows) *
                static_cast<size_t>(3 * backend.n_embd) ||
        backend.shortconv_bx.size() != static_cast<size_t>(backend.n_embd) ||
        backend.shortconv_conv_out_chunk8.size() != backend.hidden_chunk8.size()) {
      return false;
    }

    if (!measure_subprobe_bool(
            breakdown.linear_ns,
            breakdown.shortconv_in_proj_prepare_ns,
            [&]() {
              return emel::generator::detail::prepare_q8_chunk8_input(
                  backend, backend.norm_chunk8, backend.n_embd);
            }) ||
        !measure_subprobe_bool(
            breakdown.linear_ns,
            breakdown.shortconv_in_proj_ns,
            [&]() {
              return emel::generator::detail::matmul_chunk8_q8_input(
                  backend,
                  block.shortconv_in_proj,
                  backend.n_embd,
                  backend.shortconv_bcx_chunk8);
            })) {
      return false;
    }

    const size_t layer_offset = emel::generator::detail::shortconv_state_layer_offset(
        backend, layer_index);
    float * state = backend.recurrent_shortconv_cache.data() + layer_offset;
    for (int32_t row = 0; row < emel::generator::detail::k_prefill_q8_chunk8_rows; ++row) {
      const auto bcx_row = emel::generator::detail::chunk8_row_span<const float>(
          std::span<const float>(backend.shortconv_bcx_chunk8), row, 3 * backend.n_embd);
      auto conv_out_row = emel::generator::detail::chunk8_row_span<float>(
          std::span<float>(backend.shortconv_conv_out_chunk8), row, backend.n_embd);
      auto b = bcx_row.subspan(0u, static_cast<size_t>(backend.n_embd));
      auto c = bcx_row.subspan(
          static_cast<size_t>(backend.n_embd), static_cast<size_t>(backend.n_embd));
      auto x = bcx_row.subspan(
          static_cast<size_t>(2 * backend.n_embd), static_cast<size_t>(backend.n_embd));

      measure_subprobe_void(
          breakdown.misc_ns,
          breakdown.shortconv_conv_ns,
          [&]() {
            for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
              const size_t dim = static_cast<size_t>(idx);
              const float bx = b[dim] * x[dim];
              backend.shortconv_bx[dim] = bx;

              const float * kernel =
                  block.shortconv_conv.data() +
                  (dim * static_cast<size_t>(backend.shortconv_kernel_size));
              float conv_sum = bx * kernel[static_cast<size_t>(backend.shortconv_state_size)];
              for (int32_t tap = 0; tap < backend.shortconv_state_size; ++tap) {
                conv_sum +=
                    state[static_cast<size_t>(tap) * static_cast<size_t>(backend.n_embd) + dim] *
                    kernel[static_cast<size_t>(tap)];
              }

              conv_out_row[dim] = c[dim] * conv_sum;
            }
          });

      measure_subprobe_void(
          breakdown.misc_ns,
          breakdown.shortconv_state_shift_ns,
          [&]() {
            if (backend.shortconv_state_size > 1) {
              const size_t move_count =
                  static_cast<size_t>(backend.shortconv_state_size - 1) *
                  static_cast<size_t>(backend.n_embd);
              std::memmove(
                  state,
                  state + static_cast<size_t>(backend.n_embd),
                  move_count * sizeof(float));
            }
            std::memcpy(
                state + static_cast<size_t>(backend.shortconv_state_size - 1) *
                            static_cast<size_t>(backend.n_embd),
                backend.shortconv_bx.data(),
                static_cast<size_t>(backend.n_embd) * sizeof(float));
          });
    }

    if (!measure_subprobe_bool(
            breakdown.linear_ns,
            breakdown.shortconv_out_proj_prepare_ns,
            [&]() {
              return emel::generator::detail::prepare_q8_chunk8_input(
                  backend, backend.shortconv_conv_out_chunk8, backend.n_embd);
            }) ||
        !measure_subprobe_bool(
            breakdown.linear_ns,
            breakdown.shortconv_out_proj_ns,
            [&]() {
              return emel::generator::detail::matmul_chunk8_q8_input(
                         backend,
                         block.shortconv_out_proj,
                         backend.n_embd,
                         backend.projected_chunk8) &&
                  emel::generator::detail::add_chunk8_rows_in_place(
                      backend.hidden_chunk8, backend.projected_chunk8, backend.n_embd);
            })) {
      return false;
    }

    breakdown.misc_shortconv_ns =
        breakdown.shortconv_conv_ns + breakdown.shortconv_state_shift_ns;
  }

  if (!measure_subprobe_bool(
          breakdown.misc_ns,
          breakdown.misc_ffn_norm_ns,
          [&]() {
            return emel::generator::detail::rms_norm_chunk8(
                backend.hidden_chunk8,
                backend.n_embd,
                block.feed_forward_norm,
                backend.rms_epsilon,
                backend.norm_chunk8);
          })) {
    return false;
  }

  if (!measure_probe_bool(
          breakdown.linear_ns,
          [&]() {
            auto gate_chunk =
                std::span<float>(backend.gate_chunk8.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk8_rows) *
                                     static_cast<size_t>(ffn_dim));
            auto up_chunk =
                std::span<float>(backend.up_chunk8.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk8_rows) *
                                     static_cast<size_t>(ffn_dim));
            return emel::generator::detail::prepare_q8_chunk8_input(
                       backend, backend.norm_chunk8, backend.n_embd) &&
                emel::generator::detail::matmul_chunk8_q8_input(
                           backend, block.feed_forward_gate, backend.n_embd, gate_chunk) &&
                emel::generator::detail::matmul_chunk8_q8_input(
                           backend, block.feed_forward_up, backend.n_embd, up_chunk);
          })) {
    return false;
  }

  if (!measure_subprobe_bool(
          breakdown.misc_ns,
          breakdown.misc_silu_ns,
          [&]() {
            auto gate_chunk =
                std::span<float>(backend.gate_chunk8.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk8_rows) *
                                     static_cast<size_t>(ffn_dim));
            auto up_chunk =
                std::span<float>(backend.up_chunk8.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk8_rows) *
                                     static_cast<size_t>(ffn_dim));
            auto ffn_hidden_chunk =
                std::span<float>(backend.ffn_hidden_chunk8.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk8_rows) *
                                     static_cast<size_t>(ffn_dim));
            return emel::generator::detail::apply_silu_mul_chunk8(
                gate_chunk, up_chunk, ffn_dim, ffn_hidden_chunk);
          })) {
    return false;
  }

  if (!measure_probe_bool(
          breakdown.linear_ns,
          [&]() {
            auto ffn_hidden_chunk =
                std::span<float>(backend.ffn_hidden_chunk8.data(),
                                 static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk8_rows) *
                                     static_cast<size_t>(ffn_dim));
            return emel::generator::detail::prepare_q8_chunk8_input(
                       backend, ffn_hidden_chunk, ffn_dim) &&
                emel::generator::detail::matmul_chunk8_q8_input(
                           backend, block.feed_forward_down, ffn_dim, backend.projected_chunk8) &&
                emel::generator::detail::add_chunk8_rows_in_place(
                           backend.hidden_chunk8, backend.projected_chunk8, backend.n_embd);
          })) {
    return false;
  }

  return true;
}

template <emel::generator::attention_mode mode>
bool run_emel_prefill_probe_scalar(emel::generator::detail::native_backend & backend,
                                   int32_t & selected_index,
                                   float & selected_score,
                                   prefill_probe_breakdown & breakdown) {
  backend.kv_cache_tokens = 0;
  measure_probe_void(
      breakdown.misc_ns, [&]() { emel::generator::detail::reset_shortconv_cache(backend); });
  for (size_t token_index = 0; token_index < static_cast<size_t>(backend.bound_token_count);
       ++token_index) {
    const int32_t token_id = backend.bound_tokens[token_index];
    const int32_t position = backend.bound_positions[token_index];
    if (token_id < 0 || token_id >= backend.token_embedding.rows ||
        position < 0 || position >= backend.n_ctx) {
      return false;
    }
    if (!measure_probe_bool(
            breakdown.misc_ns,
            [&]() {
              return emel::generator::detail::copy_tensor_row(
                  *backend.token_embedding.tensor, token_id, backend.hidden);
            })) {
      return false;
    }
    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_emel_runtime_layer_probe_scalar<mode>(backend, layer, position, breakdown)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }
  return measure_probe_bool(
      breakdown.linear_ns,
      [&]() {
        return emel::generator::detail::compute_logits_preselected_argmax(
            backend, selected_index, selected_score);
      });
}

template <emel::generator::attention_mode mode, emel::generator::detail::chunk4_rhs_route route>
bool run_emel_prefill_probe_chunk4(emel::generator::detail::native_backend & backend,
                                   int32_t & selected_index,
                                   float & selected_score,
                                   prefill_probe_breakdown & breakdown) {
  backend.kv_cache_tokens = 0;
  measure_probe_void(
      breakdown.misc_ns, [&]() { emel::generator::detail::reset_shortconv_cache(backend); });

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  const size_t chunk_rows =
      static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk_rows);
  const size_t chunk_limit = token_count - (token_count % chunk_rows);
  if (chunk_limit == 0u) {
    return false;
  }

  for (size_t token_base = 0; token_base < chunk_limit; token_base += chunk_rows) {
    for (int32_t row = 0; row < emel::generator::detail::k_prefill_q8_chunk_rows; ++row) {
      const size_t token_index = token_base + static_cast<size_t>(row);
      const int32_t token_id = backend.bound_tokens[token_index];
      const int32_t position = backend.bound_positions[token_index];
      if (token_id < 0 || token_id >= backend.token_embedding.rows ||
          position < 0 || position >= backend.n_ctx) {
        return false;
      }
      if (!measure_probe_bool(
              breakdown.misc_ns,
              [&]() {
                return emel::generator::detail::copy_tensor_row(
                    *backend.token_embedding.tensor,
                    token_id,
                    emel::generator::detail::chunk4_row_span<float>(
                        std::span<float>(backend.hidden_chunk4), row, backend.n_embd));
              })) {
        return false;
      }
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_emel_runtime_layer_probe_chunk4<mode, route>(
              backend, layer, token_base, breakdown)) {
        return false;
      }
    }

    std::copy(
        emel::generator::detail::chunk4_row_span<const float>(
            std::span<const float>(backend.hidden_chunk4),
            emel::generator::detail::k_prefill_q8_chunk_rows - 1,
            backend.n_embd)
            .begin(),
        emel::generator::detail::chunk4_row_span<const float>(
            std::span<const float>(backend.hidden_chunk4),
            emel::generator::detail::k_prefill_q8_chunk_rows - 1,
            backend.n_embd)
            .end(),
        backend.hidden.begin());
  }

  for (size_t token_index = chunk_limit; token_index < token_count; ++token_index) {
    const int32_t token_id = backend.bound_tokens[token_index];
    const int32_t position = backend.bound_positions[token_index];
    if (token_id < 0 || token_id >= backend.token_embedding.rows ||
        position < 0 || position >= backend.n_ctx) {
      return false;
    }
    if (!measure_probe_bool(
            breakdown.misc_ns,
            [&]() {
              return emel::generator::detail::copy_tensor_row(
                  *backend.token_embedding.tensor, token_id, backend.hidden);
            })) {
      return false;
    }
    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_emel_runtime_layer_probe_scalar<mode>(backend, layer, position, breakdown)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return measure_probe_bool(
      breakdown.linear_ns,
      [&]() {
        return emel::generator::detail::compute_logits_preselected_argmax(
            backend, selected_index, selected_score);
      });
}

template <emel::generator::attention_mode mode>
bool run_emel_prefill_probe_chunk8(emel::generator::detail::native_backend & backend,
                                   int32_t & selected_index,
                                   float & selected_score,
                                   prefill_probe_breakdown & breakdown) {
  backend.kv_cache_tokens = 0;
  measure_probe_void(
      breakdown.misc_ns, [&]() { emel::generator::detail::reset_shortconv_cache(backend); });

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  const size_t chunk_rows =
      static_cast<size_t>(emel::generator::detail::k_prefill_q8_chunk8_rows);
  const size_t chunk_limit = token_count - (token_count % chunk_rows);
  if (chunk_limit == 0u) {
    return false;
  }

  for (size_t token_base = 0; token_base < chunk_limit; token_base += chunk_rows) {
    for (int32_t row = 0; row < emel::generator::detail::k_prefill_q8_chunk8_rows; ++row) {
      const size_t token_index = token_base + static_cast<size_t>(row);
      const int32_t token_id = backend.bound_tokens[token_index];
      const int32_t position = backend.bound_positions[token_index];
      if (token_id < 0 || token_id >= backend.token_embedding.rows ||
          position < 0 || position >= backend.n_ctx) {
        return false;
      }
      if (!measure_probe_bool(
              breakdown.misc_ns,
              [&]() {
                return emel::generator::detail::copy_tensor_row(
                    *backend.token_embedding.tensor,
                    token_id,
                    emel::generator::detail::chunk8_row_span<float>(
                        std::span<float>(backend.hidden_chunk8), row, backend.n_embd));
              })) {
        return false;
      }
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_emel_runtime_layer_probe_chunk8<mode>(backend, layer, token_base, breakdown)) {
        return false;
      }
    }

    std::copy(
        emel::generator::detail::chunk8_row_span<const float>(
            std::span<const float>(backend.hidden_chunk8),
            emel::generator::detail::k_prefill_q8_chunk8_rows - 1,
            backend.n_embd)
            .begin(),
        emel::generator::detail::chunk8_row_span<const float>(
            std::span<const float>(backend.hidden_chunk8),
            emel::generator::detail::k_prefill_q8_chunk8_rows - 1,
            backend.n_embd)
            .end(),
        backend.hidden.begin());
  }

  for (size_t token_index = chunk_limit; token_index < token_count; ++token_index) {
    const int32_t token_id = backend.bound_tokens[token_index];
    const int32_t position = backend.bound_positions[token_index];
    if (token_id < 0 || token_id >= backend.token_embedding.rows ||
        position < 0 || position >= backend.n_ctx) {
      return false;
    }
    if (!measure_probe_bool(
            breakdown.misc_ns,
            [&]() {
              return emel::generator::detail::copy_tensor_row(
                  *backend.token_embedding.tensor, token_id, backend.hidden);
            })) {
      return false;
    }
    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_emel_runtime_layer_probe_scalar<mode>(backend, layer, position, breakdown)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return measure_probe_bool(
      breakdown.linear_ns,
      [&]() {
        return emel::generator::detail::compute_logits_preselected_argmax(
            backend, selected_index, selected_score);
      });
}

bool measure_emel_prefill_probe(emel::generator::detail::native_backend & backend,
                                const std::vector<int32_t> & prompt_tokens,
                                const emel::generator::prefill_compute_contract contract,
                                prefill_probe_breakdown & breakdown_out,
                                int32_t & selected_index_out) {
  if (!bind_emel_prefill_probe_inputs(backend, prompt_tokens)) {
    return false;
  }

  float selected_score = 0.0f;
  switch (contract) {
    case emel::generator::prefill_compute_contract::flash_preselected_chunk8_q8_k:
      return run_emel_prefill_probe_chunk8<emel::generator::attention_mode::flash>(
          backend, selected_index_out, selected_score, breakdown_out);
    case emel::generator::prefill_compute_contract::flash_preselected_chunk4_packed_q8_0:
      return run_emel_prefill_probe_chunk4<
          emel::generator::attention_mode::flash,
          emel::generator::detail::chunk4_rhs_route::packed_q8_0>(
          backend, selected_index_out, selected_score, breakdown_out);
    case emel::generator::prefill_compute_contract::flash_preselected_chunk4_q8_k:
      return run_emel_prefill_probe_chunk4<
          emel::generator::attention_mode::flash,
          emel::generator::detail::chunk4_rhs_route::q8_k>(
          backend, selected_index_out, selected_score, breakdown_out);
    case emel::generator::prefill_compute_contract::flash_preselected_scalar:
      return run_emel_prefill_probe_scalar<emel::generator::attention_mode::flash>(
          backend, selected_index_out, selected_score, breakdown_out);
    case emel::generator::prefill_compute_contract::nonflash_preselected_chunk8_q8_k:
      return run_emel_prefill_probe_chunk8<emel::generator::attention_mode::nonflash>(
          backend, selected_index_out, selected_score, breakdown_out);
    case emel::generator::prefill_compute_contract::nonflash_preselected_chunk4_packed_q8_0:
      return run_emel_prefill_probe_chunk4<
          emel::generator::attention_mode::nonflash,
          emel::generator::detail::chunk4_rhs_route::packed_q8_0>(
          backend, selected_index_out, selected_score, breakdown_out);
    case emel::generator::prefill_compute_contract::nonflash_preselected_chunk4_q8_k:
      return run_emel_prefill_probe_chunk4<
          emel::generator::attention_mode::nonflash,
          emel::generator::detail::chunk4_rhs_route::q8_k>(
          backend, selected_index_out, selected_score, breakdown_out);
    case emel::generator::prefill_compute_contract::nonflash_preselected_scalar:
      return run_emel_prefill_probe_scalar<emel::generator::attention_mode::nonflash>(
          backend, selected_index_out, selected_score, breakdown_out);
    case emel::generator::prefill_compute_contract::flash_materialized_chunk8_q8_k:
    case emel::generator::prefill_compute_contract::flash_materialized_chunk4_packed_q8_0:
    case emel::generator::prefill_compute_contract::flash_materialized_chunk4_q8_k:
    case emel::generator::prefill_compute_contract::flash_materialized_scalar:
    case emel::generator::prefill_compute_contract::nonflash_materialized_chunk8_q8_k:
    case emel::generator::prefill_compute_contract::nonflash_materialized_chunk4_packed_q8_0:
    case emel::generator::prefill_compute_contract::nonflash_materialized_chunk4_q8_k:
    case emel::generator::prefill_compute_contract::nonflash_materialized_scalar:
    case emel::generator::prefill_compute_contract::none:
      return false;
  }
  return false;
}

bool emel_token_is_stop(const emel::model::data::vocab & vocab, const int32_t token_id) {
  return token_id == vocab.eos_id || token_id == vocab.eot_id;
}

bool run_emel_runtime_layer(emel::generator::detail::native_backend & backend,
                            const int32_t layer_index,
                            const int32_t position) {
  return emel::generator::detail::flash_attention_supported(backend, position)
      ? emel::generator::detail::run_layer_flash(backend, layer_index, position)
      : emel::generator::detail::run_layer_nonflash(backend, layer_index, position);
}

bool measure_emel_stage_probe(emel_session & session,
                              const generation_case_spec & spec,
                              emel::bench::generation_stage_probe & probe_out) {
  auto total_start = steady_clock::now();
  generation_result total_result = {};
  if (!run_emel_generate(session, spec, total_result)) {
    return false;
  }
  probe_out.emel_total_ns = elapsed_ns(total_start, steady_clock::now());

  std::vector<int32_t> prompt_tokens;
  const auto conditioning_start = steady_clock::now();
  if (!tokenize_conditioned_prompt(session, spec, prompt_tokens) || prompt_tokens.empty()) {
    return false;
  }
  probe_out.emel_conditioning_ns = elapsed_ns(conditioning_start, steady_clock::now());
  probe_out.emel_prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
  if (!inspect_emel_prefill_plan(
          session.model_data, prompt_tokens, probe_out.emel_prefill_step_size)) {
    return false;
  }
  emel::generator::prefill_compute_contract contract =
      emel::generator::prefill_compute_contract::none;
  if (!inspect_emel_prefill_contract(
          session.model_data, probe_out.emel_prompt_tokens, contract)) {
    return false;
  }
  probe_out.emel_prefill_contract = prefill_contract_name(contract);

  emel::generator::detail::native_backend backend = {};
  if (emel::generator::detail::prepare(backend, session.model_data) !=
      emel::error::cast(emel::model::loader::error::none)) {
    return false;
  }

  emel::text::renderer::sm renderer = {};
  if (!initialize_emel_renderer(session.model_data, renderer)) {
    return false;
  }

  generation_result result_out = {};
  int32_t selected_token =
      std::numeric_limits<int32_t>::min();
  const auto prefill_start = steady_clock::now();
  prefill_probe_breakdown prefill_breakdown = {};
  if (!measure_emel_prefill_probe(
          backend,
          prompt_tokens,
          contract,
          prefill_breakdown,
          selected_token)) {
    return false;
  }
  if (spec.max_tokens == 1) {
    probe_out.emel_prefill_linear_probe_ns = prefill_breakdown.linear_ns;
    probe_out.emel_prefill_attention_probe_ns = prefill_breakdown.attention_ns;
    probe_out.emel_prefill_misc_probe_ns = prefill_breakdown.misc_ns;
    probe_out.emel_prefill_misc_attention_norm_ns = prefill_breakdown.misc_attention_norm_ns;
    probe_out.emel_prefill_misc_qk_norm_ns = prefill_breakdown.misc_qk_norm_ns;
    probe_out.emel_prefill_misc_rope_ns = prefill_breakdown.misc_rope_ns;
    probe_out.emel_prefill_misc_kv_store_ns = prefill_breakdown.misc_kv_store_ns;
    probe_out.emel_prefill_misc_ctx_copy_ns = prefill_breakdown.misc_ctx_copy_ns;
    probe_out.emel_prefill_misc_shortconv_ns = prefill_breakdown.misc_shortconv_ns;
    probe_out.emel_prefill_shortconv_in_proj_ns = prefill_breakdown.shortconv_in_proj_ns;
    probe_out.emel_prefill_shortconv_in_proj_prepare_ns =
        prefill_breakdown.shortconv_in_proj_prepare_ns;
    probe_out.emel_prefill_shortconv_conv_ns = prefill_breakdown.shortconv_conv_ns;
    probe_out.emel_prefill_shortconv_state_shift_ns = prefill_breakdown.shortconv_state_shift_ns;
    probe_out.emel_prefill_shortconv_out_proj_ns = prefill_breakdown.shortconv_out_proj_ns;
    probe_out.emel_prefill_shortconv_out_proj_prepare_ns =
        prefill_breakdown.shortconv_out_proj_prepare_ns;
    probe_out.emel_prefill_misc_ffn_norm_ns = prefill_breakdown.misc_ffn_norm_ns;
    probe_out.emel_prefill_misc_silu_ns = prefill_breakdown.misc_silu_ns;
  }
  probe_out.emel_prefill_ns = elapsed_ns(prefill_start, steady_clock::now());

  for (int32_t step = 0; step < spec.max_tokens; ++step) {
    result_out.tokens_generated += 1;
    if (!append_rendered_token(renderer, selected_token, result_out)) {
      return false;
    }
    if (emel_token_is_stop(session.model_data.vocab_data, selected_token)) {
      break;
    }

    const auto decode_start = steady_clock::now();
    const int32_t position =
        static_cast<int32_t>(prompt_tokens.size()) + result_out.tokens_generated - 1;
    if (!emel::generator::detail::copy_tensor_row(
            *backend.token_embedding.tensor, selected_token, backend.hidden)) {
      return false;
    }
    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_emel_runtime_layer(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
    if (!emel::generator::detail::compute_logits(backend)) {
      return false;
    }
    const auto decode_ns = elapsed_ns(decode_start, steady_clock::now());
    selected_token = static_cast<int32_t>(select_argmax_token_from_logits(
        backend.bound_logits.data(), backend.n_vocab));
    if (step == 0) {
      probe_out.emel_first_decode_ns = decode_ns;
    } else {
      probe_out.emel_steady_decode_ns += decode_ns;
    }
  }

  if (!flush_rendered_output(renderer, result_out)) {
    return false;
  }

  probe_out.emel_unattributed_ns =
      saturating_remainder(probe_out.emel_total_ns,
                           probe_out.emel_conditioning_ns,
                           probe_out.emel_prefill_ns,
                           probe_out.emel_first_decode_ns,
                           probe_out.emel_steady_decode_ns);
  return true;
}

bool measure_reference_prefill_probe(const reference_fixture & fixture,
                                     const std::vector<llama_token> & prompt_tokens,
                                     prefill_probe_breakdown & breakdown_out) {
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
  if (run_direct_reference_decode(probe_seam, probe_ctx.get(), prompt_batch) != 0) {
    return false;
  }

  breakdown_out = probe_state.breakdown;
  return true;
}

bool measure_reference_stage_probe(const reference_fixture & fixture,
                                   const generation_case_spec & spec,
                                   emel::bench::generation_stage_probe & probe_out) {
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
  if (!tokenize_reference_prompt(fixture, spec, seam, prompt_tokens) || prompt_tokens.empty()) {
    return false;
  }
  probe_out.reference_conditioning_ns = elapsed_ns(conditioning_start, steady_clock::now());

  if (fixture.context == nullptr) {
    return false;
  }
  llama_context * ctx = fixture.context.get();
  if (!reset_reference_context(ctx)) {
    return false;
  }

  generation_result result_out = {};
  const auto prefill_start = steady_clock::now();
  llama_batch prompt_batch =
      llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
  if (run_direct_reference_decode(seam, ctx, prompt_batch) != 0) {
    return false;
  }
  probe_out.reference_prefill_ns = elapsed_ns(prefill_start, steady_clock::now());
  if (spec.max_tokens == 1) {
    prefill_probe_breakdown prefill_breakdown = {};
    if (!measure_reference_prefill_probe(fixture, prompt_tokens, prefill_breakdown)) {
      return false;
    }
    probe_out.reference_prefill_linear_probe_ns = prefill_breakdown.linear_ns;
    probe_out.reference_prefill_attention_probe_ns = prefill_breakdown.attention_ns;
    probe_out.reference_prefill_misc_probe_ns = prefill_breakdown.misc_ns;
  }

  for (int32_t step = 0; step < spec.max_tokens; ++step) {
    float * logits = read_direct_reference_logits(seam, ctx);
    if (logits == nullptr) {
      return false;
    }

    const llama_token selected =
        select_argmax_token_from_logits(logits, fixture.vocab_size);
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

  probe_out.reference_unattributed_ns =
      saturating_remainder(probe_out.reference_total_ns,
                           probe_out.reference_conditioning_ns,
                           probe_out.reference_prefill_ns,
                           probe_out.reference_first_decode_ns,
                           probe_out.reference_steady_decode_ns);
  return true;
}

bool capture_generation_stage_probe(emel_session & session,
                                    const reference_fixture & fixture,
                                    const generation_case_spec & spec) {
  emel::bench::generation_stage_probe probe = {};
  probe.name = std::string(spec.name);
  return measure_emel_stage_probe(session, spec, probe) &&
      measure_reference_stage_probe(fixture, spec, probe) &&
      (g_generation_stage_probes.push_back(std::move(probe)), true);
}

emel::bench::config generation_case_config(const emel::bench::config & cfg) {
  emel::bench::config case_cfg = cfg;
  case_cfg.iterations = read_env_u64("EMEL_BENCH_GENERATION_ITERS", 1u);
  case_cfg.runs = read_env_size("EMEL_BENCH_GENERATION_RUNS", cfg.runs);
  case_cfg.warmup_iterations = read_env_u64("EMEL_BENCH_GENERATION_WARMUP_ITERS", 0u);
  case_cfg.warmup_runs = read_env_size("EMEL_BENCH_GENERATION_WARMUP_RUNS", 0u);
  return case_cfg;
}

void prepare_emel_generation_fixture(const generation_fixture_spec & spec,
                                     prepared_emel_generation_fixture & prepared_fixture) {
  prepared_fixture.spec = &spec;
  prepared_fixture.cases = generation_cases_for_fixture(*spec.fixture, false);
  g_generation_fixture_rel = spec.fixture->fixture_rel;

  if (prepared_fixture.cases.empty()) {
    fail_bench_setup("prepare_emel_generation_fixture", "no workload manifests for fixture");
  }
  for (const generation_case_spec & generation_case : prepared_fixture.cases) {
    validate_generation_workload_fixture(*spec.fixture, generation_case);
  }

  const std::string model_path = resolve_generation_model_path(spec.fixture->fixture_rel);
  if (!prepare_emel_fixture(prepared_fixture.emel, model_path)) {
    if (prepared_fixture.emel.load.error) {
      fail_bench_setup("prepare_emel_fixture",
                       model_loader_error_name(prepared_fixture.emel.load.err).data());
    }
    if (!prepared_fixture.emel.formatter_binding.contract.empty()) {
      g_generation_formatter_contract.assign(prepared_fixture.emel.formatter_binding.contract);
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
  prepared_fixture.emel.model_data.params.n_vocab =
      static_cast<int32_t>(prepared_fixture.emel.model_data.vocab_data.n_tokens);
}

void prepare_reference_generation_fixture(const generation_fixture_spec & spec,
                                          prepared_reference_generation_fixture & prepared_fixture) {
  ensure_llama_backend_ready();

  prepared_fixture.spec = &spec;
  prepared_fixture.cases = generation_cases_for_fixture(*spec.fixture, true);
  g_generation_fixture_rel = spec.fixture->fixture_rel;

  if (prepared_fixture.cases.empty()) {
    fail_bench_setup("prepare_reference_generation_fixture", "no comparable workload manifests");
  }
  for (const generation_case_spec & generation_case : prepared_fixture.cases) {
    validate_generation_workload_fixture(*spec.fixture, generation_case);
  }

  const std::string model_path = resolve_generation_model_path(spec.fixture->fixture_rel);
  if (!prepare_reference_fixture(prepared_fixture.reference, model_path)) {
    fail_bench_setup("prepare_reference_fixture", model_path.c_str());
  }
}

void prepare_compare_generation_fixture(const generation_fixture_spec & spec,
                                        prepared_generation_fixture & prepared_fixture) {
  prepared_fixture.spec = &spec;
  prepared_fixture.cases = generation_cases_for_fixture(*spec.fixture, true);
  g_generation_fixture_rel = spec.fixture->fixture_rel;

  if (prepared_fixture.cases.empty()) {
    fail_bench_setup("prepare_compare_generation_fixture", "no comparable workload manifests");
  }
  for (const generation_case_spec & generation_case : prepared_fixture.cases) {
    validate_generation_workload_fixture(*spec.fixture, generation_case);
  }

  const std::string model_path = resolve_generation_model_path(spec.fixture->fixture_rel);
  if (!prepare_emel_fixture(prepared_fixture.emel, model_path)) {
    if (prepared_fixture.emel.load.error) {
      fail_bench_setup("prepare_emel_fixture",
                       model_loader_error_name(prepared_fixture.emel.load.err).data());
    }
    if (!prepared_fixture.emel.formatter_binding.contract.empty()) {
      g_generation_formatter_contract.assign(prepared_fixture.emel.formatter_binding.contract);
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
  prepared_fixture.emel.model_data.params.n_vocab =
      static_cast<int32_t>(prepared_fixture.emel.model_data.vocab_data.n_tokens);

  ensure_llama_backend_ready();
  if (!prepare_reference_fixture(prepared_fixture.reference, model_path)) {
    fail_bench_setup("prepare_reference_fixture", model_path.c_str());
  }
}

const std::vector<prepared_emel_generation_fixture> & maintained_emel_generation_fixtures() {
  static const std::vector<prepared_emel_generation_fixture> fixtures = [] {
    std::vector<prepared_emel_generation_fixture> prepared_fixtures = {};
    prepared_fixtures.reserve(k_emel_generation_fixtures.size());
    for (size_t fixture_index = 0u; fixture_index < k_emel_generation_fixtures.size();
         ++fixture_index) {
      const generation_fixture_spec & spec = k_emel_generation_fixtures[fixture_index];
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

const std::vector<prepared_reference_generation_fixture> & maintained_reference_generation_fixtures() {
  static const std::vector<prepared_reference_generation_fixture> fixtures = [] {
    std::vector<prepared_reference_generation_fixture> prepared_fixtures = {};
    prepared_fixtures.reserve(k_compare_generation_fixtures.size());
    for (size_t fixture_index = 0u; fixture_index < k_compare_generation_fixtures.size();
         ++fixture_index) {
      const generation_fixture_spec & spec = k_compare_generation_fixtures[fixture_index];
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

const std::vector<prepared_generation_fixture> & maintained_compare_generation_fixtures() {
  static const std::vector<prepared_generation_fixture> fixtures = [] {
    std::vector<prepared_generation_fixture> prepared_fixtures = {};
    prepared_fixtures.reserve(k_compare_generation_fixtures.size());
    for (size_t fixture_index = 0u; fixture_index < k_compare_generation_fixtures.size();
         ++fixture_index) {
      const generation_fixture_spec & spec = k_compare_generation_fixtures[fixture_index];
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

}  // namespace

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

std::uint32_t generation_runtime_contract_native_quantized_stage_count() noexcept {
  return g_generation_flash_evidence.native_quantized_stage_count;
}

std::uint32_t generation_runtime_contract_approved_dense_f32_stage_count() noexcept {
  return g_generation_flash_evidence.approved_dense_f32_stage_count;
}

std::uint32_t generation_runtime_contract_disallowed_fallback_stage_count() noexcept {
  return g_generation_flash_evidence.disallowed_fallback_stage_count;
}

std::uint32_t generation_runtime_contract_explicit_no_claim_stage_count() noexcept {
  return g_generation_flash_evidence.explicit_no_claim_stage_count;
}

std::uint64_t generation_quantized_evidence_native_q8_0_dispatch_calls() noexcept {
  return g_generation_flash_evidence.native_q8_0_dispatch_calls;
}

std::uint64_t generation_quantized_evidence_packed_q8_0_dispatch_calls() noexcept {
  return g_generation_flash_evidence.packed_q8_0_dispatch_calls;
}

std::uint64_t generation_quantized_evidence_optimized_q2_dispatch_calls() noexcept {
  return g_generation_flash_evidence.optimized_q2_dispatch_calls;
}

std::uint64_t generation_quantized_evidence_shared_q2_dispatch_calls() noexcept {
  return g_generation_flash_evidence.shared_q2_dispatch_calls;
}

std::uint64_t generation_quantized_evidence_optimized_q3_dispatch_calls() noexcept {
  return g_generation_flash_evidence.optimized_q3_dispatch_calls;
}

std::uint64_t generation_quantized_evidence_shared_q3_dispatch_calls() noexcept {
  return g_generation_flash_evidence.shared_q3_dispatch_calls;
}

std::uint64_t generation_quantized_evidence_optimized_q4_dispatch_calls() noexcept {
  return g_generation_flash_evidence.optimized_q4_dispatch_calls;
}

std::uint64_t generation_quantized_evidence_shared_q4_dispatch_calls() noexcept {
  return g_generation_flash_evidence.shared_q4_dispatch_calls;
}

std::uint64_t generation_quantized_evidence_optimized_q6_dispatch_calls() noexcept {
  return g_generation_flash_evidence.optimized_q6_dispatch_calls;
}

std::uint64_t generation_quantized_evidence_shared_q6_dispatch_calls() noexcept {
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

generation_stage_probe generation_stage_probe_at(const std::size_t index) noexcept {
  return index < g_generation_stage_probes.size()
      ? g_generation_stage_probes[index]
      : generation_stage_probe{};
}

void append_emel_generation_cases(std::vector<result> & results, const config & cfg) {
  const bool compare_lane = generation_lane_mode_current() == generation_lane_mode::compare;
  const config case_cfg = generation_case_config(cfg);

  reset_generation_flash_evidence();
  if (compare_lane) {
    const auto & fixtures = maintained_compare_generation_fixtures();
    for (size_t fixture_index = 0u; fixture_index < fixtures.size(); ++fixture_index) {
      const prepared_generation_fixture & prepared_fixture = fixtures[fixture_index];
      const generation_fixture_spec * spec = prepared_fixture.spec;
      const emel_fixture & fixture = prepared_fixture.emel;
      g_generation_fixture_rel = spec->fixture->fixture_rel;
      for (const generation_case_spec & generation_case : prepared_fixture.cases) {
        if (!generation_workload_selected(generation_case)) {
          continue;
        }
        volatile std::size_t sink = 0u;
        generation_seam_audit seam = {};
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
        prepare_emel_session(fixture, *session);
        if (!initialize_emel_session(*session, generation_case)) {
          fail_bench_setup("initialize_emel_session", generation_case.name.data());
        }
        validate_generation_formatter_contract(generation_case, session->formatter_binding.contract);

        auto fn = [&]() {
          reset_generation_seam(session->seam);
          const std::uint64_t flash_dispatch_calls_before =
              session->generator->generation_flash_attention_dispatch_calls();
          const std::uint64_t optimized_flash_dispatch_calls_before =
              session->generator->generation_optimized_flash_dispatch_calls();
          const std::uint64_t shared_flash_dispatch_calls_before =
              session->generator->generation_shared_flash_dispatch_calls();
          native_quantized_stage_count =
              session->generator->generation_native_quantized_stage_count();
          approved_dense_f32_stage_count =
              session->generator->generation_approved_dense_f32_stage_count();
          disallowed_fallback_stage_count =
              session->generator->generation_disallowed_fallback_stage_count();
          explicit_no_claim_stage_count =
              session->generator->generation_explicit_no_claim_stage_count();
          const std::uint64_t native_q8_0_dispatch_calls_before =
              session->generator->generation_native_q8_0_dispatch_calls();
          const std::uint64_t packed_q8_0_dispatch_calls_before =
              session->generator->generation_packed_q8_0_dispatch_calls();
          const std::uint64_t optimized_q2_dispatch_calls_before =
              session->generator->generation_optimized_q2_dispatch_calls();
          const std::uint64_t shared_q2_dispatch_calls_before =
              session->generator->generation_shared_q2_dispatch_calls();
          const std::uint64_t optimized_q3_dispatch_calls_before =
              session->generator->generation_optimized_q3_dispatch_calls();
          const std::uint64_t shared_q3_dispatch_calls_before =
              session->generator->generation_shared_q3_dispatch_calls();
          const std::uint64_t optimized_q4_dispatch_calls_before =
              session->generator->generation_optimized_q4_dispatch_calls();
          const std::uint64_t shared_q4_dispatch_calls_before =
              session->generator->generation_shared_q4_dispatch_calls();
          const std::uint64_t optimized_q6_dispatch_calls_before =
              session->generator->generation_optimized_q6_dispatch_calls();
          const std::uint64_t shared_q6_dispatch_calls_before =
              session->generator->generation_shared_q6_dispatch_calls();

          generation_result generated{};
          if (!run_emel_generate(*session, generation_case, generated)) {
            fail_bench_setup("run_emel_generate", generation_case.name.data());
          }
          latest_generated = generated;
          const std::uint64_t flash_dispatch_calls_after =
              session->generator->generation_flash_attention_dispatch_calls();
          const std::uint64_t optimized_flash_dispatch_calls_after =
              session->generator->generation_optimized_flash_dispatch_calls();
          const std::uint64_t shared_flash_dispatch_calls_after =
              session->generator->generation_shared_flash_dispatch_calls();
          const std::uint64_t native_q8_0_dispatch_calls_after =
              session->generator->generation_native_q8_0_dispatch_calls();
          const std::uint64_t packed_q8_0_dispatch_calls_after =
              session->generator->generation_packed_q8_0_dispatch_calls();
          const std::uint64_t optimized_q2_dispatch_calls_after =
              session->generator->generation_optimized_q2_dispatch_calls();
          const std::uint64_t shared_q2_dispatch_calls_after =
              session->generator->generation_shared_q2_dispatch_calls();
          const std::uint64_t optimized_q3_dispatch_calls_after =
              session->generator->generation_optimized_q3_dispatch_calls();
          const std::uint64_t shared_q3_dispatch_calls_after =
              session->generator->generation_shared_q3_dispatch_calls();
          const std::uint64_t optimized_q4_dispatch_calls_after =
              session->generator->generation_optimized_q4_dispatch_calls();
          const std::uint64_t shared_q4_dispatch_calls_after =
              session->generator->generation_shared_q4_dispatch_calls();
          const std::uint64_t optimized_q6_dispatch_calls_after =
              session->generator->generation_optimized_q6_dispatch_calls();
          const std::uint64_t shared_q6_dispatch_calls_after =
              session->generator->generation_shared_q6_dispatch_calls();
          seam = session->seam;
          flash_dispatch_calls = flash_dispatch_calls_after - flash_dispatch_calls_before;
          optimized_flash_dispatch_calls =
              optimized_flash_dispatch_calls_after - optimized_flash_dispatch_calls_before;
          shared_flash_dispatch_calls =
              shared_flash_dispatch_calls_after - shared_flash_dispatch_calls_before;
          native_q8_0_dispatch_calls =
              native_q8_0_dispatch_calls_after - native_q8_0_dispatch_calls_before;
          packed_q8_0_dispatch_calls =
              packed_q8_0_dispatch_calls_after - packed_q8_0_dispatch_calls_before;
          optimized_q2_dispatch_calls =
              optimized_q2_dispatch_calls_after - optimized_q2_dispatch_calls_before;
          shared_q2_dispatch_calls =
              shared_q2_dispatch_calls_after - shared_q2_dispatch_calls_before;
          optimized_q3_dispatch_calls =
              optimized_q3_dispatch_calls_after - optimized_q3_dispatch_calls_before;
          shared_q3_dispatch_calls =
              shared_q3_dispatch_calls_after - shared_q3_dispatch_calls_before;
          optimized_q4_dispatch_calls =
              optimized_q4_dispatch_calls_after - optimized_q4_dispatch_calls_before;
          shared_q4_dispatch_calls =
              shared_q4_dispatch_calls_after - shared_q4_dispatch_calls_before;
          optimized_q6_dispatch_calls =
              optimized_q6_dispatch_calls_after - optimized_q6_dispatch_calls_before;
          shared_q6_dispatch_calls =
              shared_q6_dispatch_calls_after - shared_q6_dispatch_calls_before;
          sink ^= generated.output_length;
        };

        results.push_back(measure_case(generation_case.name.data(), case_cfg, fn));
        result & compare_record = results.back();
        compare_record.compare_group = generation_case.manifest.compare_group;
        compare_record.lane = "emel";
        compare_record.backend_id = "emel.generator";
        compare_record.backend_language = "cpp";
        compare_record.workload_id = generation_case.manifest.id;
        compare_record.workload_manifest_path = generation_case.manifest.workload_manifest_path;
        compare_record.comparison_mode = generation_case.manifest.comparison_mode;
        compare_record.model_id = generation_case.manifest.fixture_name;
        compare_record.fixture_id = generation_case.manifest.fixture_rel;
        compare_record.prompt_fixture_id = generation_case.manifest.prompt_fixture_id;
        compare_record.prompt_fixture_path = generation_case.manifest.prompt_fixture_path;
        compare_record.prompt_id = generation_case.manifest.prompt_id;
        compare_record.formatter_mode = generation_case.manifest.formatter_mode;
        compare_record.formatter_contract = generation_case.manifest.formatter_contract;
        compare_record.sampling_id = generation_case.manifest.sampling_id;
        compare_record.stop_id = generation_case.manifest.stop_id;
        compare_record.seed = generation_case.manifest.seed;
        compare_record.max_output_tokens = generation_case.manifest.max_output_tokens;
        compare_record.comparable = generation_case.manifest.comparable;
        compare_record.output_tokens = static_cast<std::uint64_t>(latest_generated.tokens_generated);
        compare_record.output_bytes = static_cast<std::uint64_t>(latest_generated.output_length);
        compare_record.output_checksum = checksum_bytes(
          reinterpret_cast<const std::uint8_t *>(latest_generated.output.data()),
          latest_generated.output_length);
        compare_record.output_text.assign(latest_generated.output.data(), latest_generated.output_length);
        compare_record.note = generation_case.manifest.comparability_note;
        if (spec->fixture->current_publication &&
            generation_case.name == k_generation_case_name) {
          g_generation_architecture_contract.assign(
              emel::model::architecture_name_view(session->model_data));
          g_generation_formatter_contract.assign(session->formatter_binding.contract);
          g_generation_flash_evidence.ready = true;
          g_generation_flash_evidence.flash_dispatch_calls = flash_dispatch_calls;
          g_generation_flash_evidence.optimized_flash_dispatch_calls =
              optimized_flash_dispatch_calls;
          g_generation_flash_evidence.shared_flash_dispatch_calls = shared_flash_dispatch_calls;
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
          g_generation_flash_evidence.shared_q2_dispatch_calls = shared_q2_dispatch_calls;
          g_generation_flash_evidence.optimized_q3_dispatch_calls =
              optimized_q3_dispatch_calls;
          g_generation_flash_evidence.shared_q3_dispatch_calls = shared_q3_dispatch_calls;
          g_generation_flash_evidence.optimized_q4_dispatch_calls =
              optimized_q4_dispatch_calls;
          g_generation_flash_evidence.shared_q4_dispatch_calls = shared_q4_dispatch_calls;
          g_generation_flash_evidence.optimized_q6_dispatch_calls =
              optimized_q6_dispatch_calls;
          g_generation_flash_evidence.shared_q6_dispatch_calls = shared_q6_dispatch_calls;
          g_generation_flash_evidence.seam = seam;
        }
        if (generation_seam_audit_enabled()) {
          print_generation_seam_audit("emel", seam);
          verify_emel_generation_seam(seam);
        }
        if (spec->fixture->current_publication &&
            !capture_generation_stage_probe(*session, prepared_fixture.reference, generation_case)) {
          fail_bench_setup("capture_generation_stage_probe", generation_case.name.data());
        }
        static_cast<void>(sink);
      }
    }
    return;
  }

  const auto & fixtures = maintained_emel_generation_fixtures();
  for (const prepared_emel_generation_fixture & prepared_fixture : fixtures) {
    const generation_fixture_spec * spec = prepared_fixture.spec;
    const emel_fixture & fixture = prepared_fixture.emel;
    g_generation_fixture_rel = spec->fixture->fixture_rel;
    for (const generation_case_spec & generation_case : prepared_fixture.cases) {
      if (!generation_workload_selected(generation_case)) {
        continue;
      }
      volatile std::size_t sink = 0u;
      generation_seam_audit seam = {};
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
      prepare_emel_session(fixture, *session);
      if (!initialize_emel_session(*session, generation_case)) {
        fail_bench_setup("initialize_emel_session", generation_case.name.data());
      }
      validate_generation_formatter_contract(generation_case, session->formatter_binding.contract);

      auto fn = [&]() {
        reset_generation_seam(session->seam);
        const std::uint64_t flash_dispatch_calls_before =
            session->generator->generation_flash_attention_dispatch_calls();
        const std::uint64_t optimized_flash_dispatch_calls_before =
            session->generator->generation_optimized_flash_dispatch_calls();
        const std::uint64_t shared_flash_dispatch_calls_before =
            session->generator->generation_shared_flash_dispatch_calls();
        native_quantized_stage_count =
            session->generator->generation_native_quantized_stage_count();
        approved_dense_f32_stage_count =
            session->generator->generation_approved_dense_f32_stage_count();
        disallowed_fallback_stage_count =
            session->generator->generation_disallowed_fallback_stage_count();
        explicit_no_claim_stage_count =
            session->generator->generation_explicit_no_claim_stage_count();
        const std::uint64_t native_q8_0_dispatch_calls_before =
            session->generator->generation_native_q8_0_dispatch_calls();
        const std::uint64_t packed_q8_0_dispatch_calls_before =
            session->generator->generation_packed_q8_0_dispatch_calls();
        const std::uint64_t optimized_q2_dispatch_calls_before =
            session->generator->generation_optimized_q2_dispatch_calls();
        const std::uint64_t shared_q2_dispatch_calls_before =
            session->generator->generation_shared_q2_dispatch_calls();
        const std::uint64_t optimized_q3_dispatch_calls_before =
            session->generator->generation_optimized_q3_dispatch_calls();
        const std::uint64_t shared_q3_dispatch_calls_before =
            session->generator->generation_shared_q3_dispatch_calls();
        const std::uint64_t optimized_q4_dispatch_calls_before =
            session->generator->generation_optimized_q4_dispatch_calls();
        const std::uint64_t shared_q4_dispatch_calls_before =
            session->generator->generation_shared_q4_dispatch_calls();
        const std::uint64_t optimized_q6_dispatch_calls_before =
            session->generator->generation_optimized_q6_dispatch_calls();
        const std::uint64_t shared_q6_dispatch_calls_before =
            session->generator->generation_shared_q6_dispatch_calls();

        generation_result generated{};
        if (!run_emel_generate(*session, generation_case, generated)) {
          fail_bench_setup("run_emel_generate", generation_case.name.data());
        }
        latest_generated = generated;
        const std::uint64_t flash_dispatch_calls_after =
            session->generator->generation_flash_attention_dispatch_calls();
        const std::uint64_t optimized_flash_dispatch_calls_after =
            session->generator->generation_optimized_flash_dispatch_calls();
        const std::uint64_t shared_flash_dispatch_calls_after =
            session->generator->generation_shared_flash_dispatch_calls();
        const std::uint64_t native_q8_0_dispatch_calls_after =
            session->generator->generation_native_q8_0_dispatch_calls();
        const std::uint64_t packed_q8_0_dispatch_calls_after =
            session->generator->generation_packed_q8_0_dispatch_calls();
        const std::uint64_t optimized_q2_dispatch_calls_after =
            session->generator->generation_optimized_q2_dispatch_calls();
        const std::uint64_t shared_q2_dispatch_calls_after =
            session->generator->generation_shared_q2_dispatch_calls();
        const std::uint64_t optimized_q3_dispatch_calls_after =
            session->generator->generation_optimized_q3_dispatch_calls();
        const std::uint64_t shared_q3_dispatch_calls_after =
            session->generator->generation_shared_q3_dispatch_calls();
        const std::uint64_t optimized_q4_dispatch_calls_after =
            session->generator->generation_optimized_q4_dispatch_calls();
        const std::uint64_t shared_q4_dispatch_calls_after =
            session->generator->generation_shared_q4_dispatch_calls();
        const std::uint64_t optimized_q6_dispatch_calls_after =
            session->generator->generation_optimized_q6_dispatch_calls();
        const std::uint64_t shared_q6_dispatch_calls_after =
            session->generator->generation_shared_q6_dispatch_calls();
        seam = session->seam;
        flash_dispatch_calls = flash_dispatch_calls_after - flash_dispatch_calls_before;
        optimized_flash_dispatch_calls =
            optimized_flash_dispatch_calls_after - optimized_flash_dispatch_calls_before;
        shared_flash_dispatch_calls =
            shared_flash_dispatch_calls_after - shared_flash_dispatch_calls_before;
        native_q8_0_dispatch_calls =
            native_q8_0_dispatch_calls_after - native_q8_0_dispatch_calls_before;
        packed_q8_0_dispatch_calls =
            packed_q8_0_dispatch_calls_after - packed_q8_0_dispatch_calls_before;
        optimized_q2_dispatch_calls =
            optimized_q2_dispatch_calls_after - optimized_q2_dispatch_calls_before;
        shared_q2_dispatch_calls =
            shared_q2_dispatch_calls_after - shared_q2_dispatch_calls_before;
        optimized_q3_dispatch_calls =
            optimized_q3_dispatch_calls_after - optimized_q3_dispatch_calls_before;
        shared_q3_dispatch_calls =
            shared_q3_dispatch_calls_after - shared_q3_dispatch_calls_before;
        optimized_q4_dispatch_calls =
            optimized_q4_dispatch_calls_after - optimized_q4_dispatch_calls_before;
        shared_q4_dispatch_calls =
            shared_q4_dispatch_calls_after - shared_q4_dispatch_calls_before;
        optimized_q6_dispatch_calls =
            optimized_q6_dispatch_calls_after - optimized_q6_dispatch_calls_before;
        shared_q6_dispatch_calls =
            shared_q6_dispatch_calls_after - shared_q6_dispatch_calls_before;
        sink ^= generated.output_length;
      };

      results.push_back(measure_case(generation_case.name.data(), case_cfg, fn));
      result & compare_record = results.back();
      compare_record.compare_group = generation_case.manifest.compare_group;
      compare_record.lane = "emel";
      compare_record.backend_id = "emel.generator";
      compare_record.backend_language = "cpp";
      compare_record.workload_id = generation_case.manifest.id;
      compare_record.workload_manifest_path = generation_case.manifest.workload_manifest_path;
      compare_record.comparison_mode = generation_case.manifest.comparison_mode;
      compare_record.model_id = generation_case.manifest.fixture_name;
      compare_record.fixture_id = generation_case.manifest.fixture_rel;
      compare_record.prompt_fixture_id = generation_case.manifest.prompt_fixture_id;
      compare_record.prompt_fixture_path = generation_case.manifest.prompt_fixture_path;
      compare_record.prompt_id = generation_case.manifest.prompt_id;
      compare_record.formatter_mode = generation_case.manifest.formatter_mode;
      compare_record.formatter_contract = generation_case.manifest.formatter_contract;
      compare_record.sampling_id = generation_case.manifest.sampling_id;
      compare_record.stop_id = generation_case.manifest.stop_id;
      compare_record.seed = generation_case.manifest.seed;
      compare_record.max_output_tokens = generation_case.manifest.max_output_tokens;
      compare_record.comparable = generation_case.manifest.comparable;
      compare_record.output_tokens = static_cast<std::uint64_t>(latest_generated.tokens_generated);
      compare_record.output_bytes = static_cast<std::uint64_t>(latest_generated.output_length);
      compare_record.output_checksum = checksum_bytes(
        reinterpret_cast<const std::uint8_t *>(latest_generated.output.data()),
        latest_generated.output_length);
      compare_record.output_text.assign(latest_generated.output.data(), latest_generated.output_length);
      compare_record.note = generation_case.manifest.comparability_note;
      if (spec->fixture->current_publication &&
          generation_case.name == k_generation_case_name) {
        g_generation_architecture_contract.assign(
            emel::model::architecture_name_view(session->model_data));
        g_generation_formatter_contract.assign(session->formatter_binding.contract);
        g_generation_flash_evidence.ready = true;
        g_generation_flash_evidence.flash_dispatch_calls = flash_dispatch_calls;
        g_generation_flash_evidence.optimized_flash_dispatch_calls =
            optimized_flash_dispatch_calls;
        g_generation_flash_evidence.shared_flash_dispatch_calls = shared_flash_dispatch_calls;
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
        g_generation_flash_evidence.shared_q2_dispatch_calls = shared_q2_dispatch_calls;
        g_generation_flash_evidence.optimized_q3_dispatch_calls =
            optimized_q3_dispatch_calls;
        g_generation_flash_evidence.shared_q3_dispatch_calls = shared_q3_dispatch_calls;
        g_generation_flash_evidence.optimized_q4_dispatch_calls =
            optimized_q4_dispatch_calls;
        g_generation_flash_evidence.shared_q4_dispatch_calls = shared_q4_dispatch_calls;
        g_generation_flash_evidence.optimized_q6_dispatch_calls =
            optimized_q6_dispatch_calls;
        g_generation_flash_evidence.shared_q6_dispatch_calls = shared_q6_dispatch_calls;
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

void append_reference_generation_cases(std::vector<result> & results, const config & cfg) {
  const bool compare_lane = generation_lane_mode_current() == generation_lane_mode::compare;
  const config case_cfg = generation_case_config(cfg);

  if (compare_lane) {
    const auto & fixtures = maintained_compare_generation_fixtures();
    for (const prepared_generation_fixture & prepared_fixture : fixtures) {
      const generation_fixture_spec * spec = prepared_fixture.spec;
      const reference_fixture & fixture = prepared_fixture.reference;
      g_generation_fixture_rel = spec->fixture->fixture_rel;
      for (const generation_case_spec & generation_case : prepared_fixture.cases) {
        if (!generation_workload_selected(generation_case)) {
          continue;
        }
        volatile std::size_t sink = 0u;
        generation_seam_audit seam = {};
        generation_result latest_generated = {};
        validate_generation_formatter_contract(generation_case, fixture.formatter.contract);
        auto fn = [&]() {
          reset_generation_seam(seam);
          generation_result generated{};
          // Keep the reference path honest with EMEL's timed generate request:
          // formatting and tokenization stay inside the measured lambda here too.
          if (!run_reference_generate(fixture, generation_case, seam, generated)) {
            fail_bench_setup("run_reference_generate", generation_case.name.data());
          }
          latest_generated = generated;
          sink ^= generated.output_length;
        };

        results.push_back(measure_case(generation_case.name.data(), case_cfg, fn));
        result & compare_record = results.back();
        compare_record.compare_group = generation_case.manifest.compare_group;
        compare_record.lane = "reference";
        compare_record.backend_id = "cpp.reference.llama_cpp";
        compare_record.backend_language = "cpp";
        compare_record.workload_id = generation_case.manifest.id;
        compare_record.workload_manifest_path = generation_case.manifest.workload_manifest_path;
        compare_record.comparison_mode = generation_case.manifest.comparison_mode;
        compare_record.model_id = generation_case.manifest.fixture_name;
        compare_record.fixture_id = generation_case.manifest.fixture_rel;
        compare_record.prompt_fixture_id = generation_case.manifest.prompt_fixture_id;
        compare_record.prompt_fixture_path = generation_case.manifest.prompt_fixture_path;
        compare_record.prompt_id = generation_case.manifest.prompt_id;
        compare_record.formatter_mode = generation_case.manifest.formatter_mode;
        compare_record.formatter_contract = generation_case.manifest.formatter_contract;
        compare_record.sampling_id = generation_case.manifest.sampling_id;
        compare_record.stop_id = generation_case.manifest.stop_id;
        compare_record.seed = generation_case.manifest.seed;
        compare_record.max_output_tokens = generation_case.manifest.max_output_tokens;
        compare_record.comparable = generation_case.manifest.comparable;
        compare_record.output_tokens = static_cast<std::uint64_t>(latest_generated.tokens_generated);
        compare_record.output_bytes = static_cast<std::uint64_t>(latest_generated.output_length);
        compare_record.output_checksum = checksum_bytes(
          reinterpret_cast<const std::uint8_t *>(latest_generated.output.data()),
          latest_generated.output_length);
        compare_record.output_text.assign(latest_generated.output.data(), latest_generated.output_length);
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

  const auto & fixtures = maintained_reference_generation_fixtures();
  for (const prepared_reference_generation_fixture & prepared_fixture : fixtures) {
    const generation_fixture_spec * spec = prepared_fixture.spec;
    const reference_fixture & fixture = prepared_fixture.reference;
    g_generation_fixture_rel = spec->fixture->fixture_rel;
    for (const generation_case_spec & generation_case : prepared_fixture.cases) {
      if (!generation_workload_selected(generation_case)) {
        continue;
      }
      volatile std::size_t sink = 0u;
      generation_seam_audit seam = {};
      generation_result latest_generated = {};
      validate_generation_formatter_contract(generation_case, fixture.formatter.contract);
      auto fn = [&]() {
        reset_generation_seam(seam);
        generation_result generated{};
        // Keep the reference path honest with EMEL's timed generate request:
        // formatting and tokenization stay inside the measured lambda here too.
        if (!run_reference_generate(fixture, generation_case, seam, generated)) {
          fail_bench_setup("run_reference_generate", generation_case.name.data());
        }
        latest_generated = generated;
        sink ^= generated.output_length;
      };

      results.push_back(measure_case(generation_case.name.data(), case_cfg, fn));
      result & compare_record = results.back();
      compare_record.compare_group = generation_case.manifest.compare_group;
      compare_record.lane = "reference";
      compare_record.backend_id = "cpp.reference.llama_cpp";
      compare_record.backend_language = "cpp";
      compare_record.workload_id = generation_case.manifest.id;
      compare_record.workload_manifest_path = generation_case.manifest.workload_manifest_path;
      compare_record.comparison_mode = generation_case.manifest.comparison_mode;
      compare_record.model_id = generation_case.manifest.fixture_name;
      compare_record.fixture_id = generation_case.manifest.fixture_rel;
      compare_record.prompt_fixture_id = generation_case.manifest.prompt_fixture_id;
      compare_record.prompt_fixture_path = generation_case.manifest.prompt_fixture_path;
      compare_record.prompt_id = generation_case.manifest.prompt_id;
      compare_record.formatter_mode = generation_case.manifest.formatter_mode;
      compare_record.formatter_contract = generation_case.manifest.formatter_contract;
      compare_record.sampling_id = generation_case.manifest.sampling_id;
      compare_record.stop_id = generation_case.manifest.stop_id;
      compare_record.seed = generation_case.manifest.seed;
      compare_record.max_output_tokens = generation_case.manifest.max_output_tokens;
      compare_record.comparable = generation_case.manifest.comparable;
      compare_record.output_tokens = static_cast<std::uint64_t>(latest_generated.tokens_generated);
      compare_record.output_bytes = static_cast<std::uint64_t>(latest_generated.output_length);
      compare_record.output_checksum = checksum_bytes(
        reinterpret_cast<const std::uint8_t *>(latest_generated.output.data()),
        latest_generated.output_length);
      compare_record.output_text.assign(latest_generated.output.data(), latest_generated.output_length);
      compare_record.note = generation_case.manifest.comparability_note;
      if (generation_seam_audit_enabled()) {
        print_generation_seam_audit("reference", seam);
        verify_reference_generation_seam(seam);
      }
      static_cast<void>(sink);
    }
  }
}

}  // namespace emel::bench
