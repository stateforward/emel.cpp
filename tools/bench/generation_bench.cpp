#include "bench_cases.hpp"
#include "../generation_formatter_contract.hpp"
#include "../generation_fixture_registry.hpp"

#include <algorithm>
#include <array>
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
#include <vector>

#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/errors.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/generator/errors.hpp"
#include "emel/generator/events.hpp"
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
#include "emel/text/tokenizer/sm.hpp"

#include "ggml.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-memory.h"
#include "llama-vocab.h"

namespace {

constexpr size_t k_generation_output_capacity = 65536u;

struct generation_case_spec {
  std::string_view name = {};
  std::string_view prompt = {};
  int32_t max_tokens = 0;
};

struct generation_fixture_spec {
  const emel::tools::generation_fixture_registry::maintained_fixture * fixture = nullptr;
  std::array<generation_case_spec, 4> cases = {};
};

constexpr generation_case_spec k_short_generation_case = {
    .name = emel::bench::k_generation_case_name,
    .prompt = "hello",
    .max_tokens = 1,
};

constexpr generation_case_spec k_generation_10_case = {
    .name = emel::bench::k_generation_10_case_name,
    .prompt = "hello",
    .max_tokens = 10,
};

constexpr generation_case_spec k_generation_100_case = {
    .name = emel::bench::k_generation_100_case_name,
    .prompt = "hello",
    .max_tokens = 100,
};

constexpr generation_case_spec k_generation_1000_case = {
    .name = emel::bench::k_generation_1000_case_name,
    .prompt = "hello",
    .max_tokens = 1000,
};

constexpr generation_case_spec k_qwen3_short_generation_case = {
    .name = emel::bench::k_generation_qwen3_case_name,
    .prompt = "hello",
    .max_tokens = 1,
};

constexpr generation_case_spec k_qwen3_generation_10_case = {
    .name = emel::bench::k_generation_qwen3_10_case_name,
    .prompt = "hello",
    .max_tokens = 10,
};

constexpr generation_case_spec k_qwen3_generation_100_case = {
    .name = emel::bench::k_generation_qwen3_100_case_name,
    .prompt = "hello",
    .max_tokens = 100,
};

constexpr generation_case_spec k_qwen3_generation_1000_case = {
    .name = emel::bench::k_generation_qwen3_1000_case_name,
    .prompt = "hello",
    .max_tokens = 1000,
};

constexpr generation_fixture_spec k_qwen3_generation_fixture = {
    .fixture = &emel::tools::generation_fixture_registry::k_qwen3_generation_fixture,
    .cases = {
        k_qwen3_short_generation_case,
        k_qwen3_generation_10_case,
        k_qwen3_generation_100_case,
        k_qwen3_generation_1000_case,
    },
};

constexpr generation_fixture_spec k_lfm2_generation_fixture = {
    .fixture = &emel::tools::generation_fixture_registry::k_lfm2_generation_fixture,
    .cases = {
        k_short_generation_case,
        k_generation_10_case,
        k_generation_100_case,
        k_generation_1000_case,
    },
};

constexpr std::array<generation_fixture_spec, 2> k_generation_fixtures = {
    k_qwen3_generation_fixture,
    k_lfm2_generation_fixture,
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

int32_t find_generation_fallback_token_id(const emel::model::data::vocab & vocab) {
  constexpr std::array<std::string_view, 4> k_preferred_tokens = {
      "hello",
      "world",
      "Hello",
      "!",
  };

  for (const std::string_view preferred : k_preferred_tokens) {
    for (uint32_t token_id = 0u; token_id < vocab.n_tokens; ++token_id) {
      if (vocab_token_view(vocab, static_cast<int32_t>(token_id)) == preferred) {
        return static_cast<int32_t>(token_id);
      }
    }
  }

  const int32_t normal_type = emel::model::detail::k_token_type_normal;
  for (uint32_t token_id = 0u; token_id < vocab.n_tokens; ++token_id) {
    const auto & entry = vocab.entries[token_id];
    const std::string_view piece = vocab_token_view(vocab, static_cast<int32_t>(token_id));
    if (entry.type == normal_type && is_printable_ascii_token(piece)) {
      return static_cast<int32_t>(token_id);
    }
  }

  return vocab.eos_id >= 0 ? vocab.eos_id : 0;
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
  llama_model_ptr reference_model = {nullptr, llama_model_free};
  const llama_vocab * reference_vocab = nullptr;
  int32_t reference_vocab_size = 0;
  int32_t fallback_token_id = 0;
  emel::tools::generation_formatter_contract::formatter_binding formatter_binding = {};
};

struct generation_seam_audit {
  int32_t emel_reference_decode_calls = 0;
  int32_t emel_reference_logits_calls = 0;
  int32_t direct_reference_decode_calls = 0;
  int32_t direct_reference_logits_calls = 0;
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
  emel_fixture fixture = {};
};

emel::model::detail::kv_binding kv_binding_from_fixture(const emel_fixture & fixture) {
  return emel::model::detail::kv_binding{
      .arena = std::span<const uint8_t>{fixture.kv_arena.data(), fixture.kv_arena.size()},
      .entries = std::span<const emel::gguf::loader::kv_entry>{fixture.kv_entries.data(),
                                                               fixture.kv_entries.size()},
  };
}

generation_flash_evidence_state g_generation_flash_evidence = {};
std::string g_generation_formatter_contract = {};
std::string g_generation_architecture_contract = {};
std::string_view g_generation_fixture_rel = {};

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

void reset_gguf_capture(emel_fixture & fixture) { fixture.gguf = {}; }
void reset_weight_capture(emel_fixture & fixture) { fixture.weight = {}; }
void reset_load_capture(emel_fixture & fixture) { fixture.load = {}; }
void reset_initialize_capture(emel_session & session) { session.initialize = {}; }
void reset_generation_capture(emel_session & session) { session.generation = {}; }

void on_probe_done(void * owner, const emel::gguf::loader::events::probe_done & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.gguf.probe_done = true;
  fixture.gguf.probe_error = false;
  fixture.gguf.requirements = ev.requirements_out;
}

void on_probe_error(void * owner, const emel::gguf::loader::events::probe_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.gguf.probe_error = true;
  fixture.gguf.err = ev.err;
}

void on_bind_done(void * owner, const emel::gguf::loader::events::bind_done &) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.gguf.bind_done = true;
  fixture.gguf.bind_error = false;
}

void on_bind_error(void * owner, const emel::gguf::loader::events::bind_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.gguf.bind_error = true;
  fixture.gguf.err = ev.err;
}

void on_parse_done(void * owner, const emel::gguf::loader::events::parse_done &) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.gguf.parse_done = true;
  fixture.gguf.parse_error = false;
}

void on_parse_error(void * owner, const emel::gguf::loader::events::parse_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.gguf.parse_error = true;
  fixture.gguf.err = ev.err;
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
               "reference_decode_calls=%d reference_logits_calls=%d\n",
               label,
               seam.emel_reference_decode_calls,
               seam.emel_reference_logits_calls,
               seam.direct_reference_decode_calls,
               seam.direct_reference_logits_calls);
}

void verify_emel_generation_seam(const generation_seam_audit & seam) {
  if (seam.emel_reference_decode_calls != 0 || seam.emel_reference_logits_calls != 0 ||
      seam.direct_reference_decode_calls != 0 || seam.direct_reference_logits_calls != 0) {
    fail_bench_setup("generation seam audit", "EMEL benchmark path touched reference decode seam");
  }
}

void verify_reference_generation_seam(const generation_seam_audit & seam) {
  if (seam.emel_reference_decode_calls != 0 || seam.emel_reference_logits_calls != 0 ||
      seam.direct_reference_decode_calls <= 0 || seam.direct_reference_logits_calls <= 0) {
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

std::string_view kv_key_view(const emel_fixture & fixture,
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

std::span<const uint8_t> kv_value_view(const emel_fixture & fixture,
                                       const emel::gguf::loader::kv_entry & entry) {
  if (static_cast<size_t>(entry.value_offset) + static_cast<size_t>(entry.value_length) >
      fixture.kv_arena.size()) {
    return {};
  }

  return std::span<const uint8_t>{fixture.kv_arena.data() + entry.value_offset, entry.value_length};
}

const emel::gguf::loader::kv_entry * find_kv_entry(const emel_fixture & fixture,
                                                   const std::string_view key) {
  for (const auto & entry : fixture.kv_entries) {
    if (kv_key_view(fixture, entry) == key) {
      return &entry;
    }
  }
  return nullptr;
}

bool decode_string_value(const emel_fixture & fixture,
                         const emel::gguf::loader::kv_entry & entry,
                         std::string_view & value_out);

emel::tools::generation_formatter_contract::formatter_binding
resolve_fixture_formatter_binding(const emel_fixture & fixture) {
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

bool decode_integer_value(const emel_fixture & fixture,
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

bool decode_float_value(const emel_fixture & fixture,
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

bool decode_bool_value(const emel_fixture & fixture,
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

bool decode_string_value(const emel_fixture & fixture,
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

bool decode_string_array_count(const emel_fixture & fixture,
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

bool decode_integer_array_first_nonzero(const emel_fixture & fixture,
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
  const auto * architecture_entry = find_kv_entry(fixture, "general.architecture");
  if (architecture_entry == nullptr) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  std::string_view architecture = {};
  if (!decode_string_value(fixture, *architecture_entry, architecture) ||
      architecture.size() >= model_data.architecture_name.size()) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  copy_name(model_data.architecture_name, architecture);
  const bool is_llama = architecture == "llama";
  const bool is_qwen3 = architecture == "qwen3";
  const bool is_lfm2 = emel::model::is_lfm2_execution_architecture(architecture);
  if (!is_llama && !is_qwen3 && !is_lfm2) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const auto assign_i32 = [&](const std::string_view key, int32_t & field) {
    const auto * entry = find_kv_entry(fixture, key);
    if (entry == nullptr) {
      return true;
    }

    uint64_t value = 0u;
    if (!decode_integer_value(fixture, *entry, value) ||
        value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }

    field = static_cast<int32_t>(value);
    return true;
  };

  const auto assign_f32 = [&](const std::string_view key, float & field) {
    const auto * entry = find_kv_entry(fixture, key);
    if (entry == nullptr) {
      return true;
    }

    float value = 0.0f;
    if (!decode_float_value(fixture, *entry, value)) {
      return false;
    }

    field = value;
    return true;
  };

  const auto assign_bool = [&](const std::string_view key, bool & field) {
    const auto * entry = find_kv_entry(fixture, key);
    if (entry == nullptr) {
      return true;
    }

    bool value = false;
    if (!decode_bool_value(fixture, *entry, value)) {
      return false;
    }

    field = value;
    return true;
  };

  if (is_llama) {
    if (!assign_i32("llama.context_length", model_data.params.n_ctx) ||
        !assign_i32("llama.embedding_length", model_data.params.n_embd) ||
        !assign_i32("llama.embedding_length_out", model_data.params.n_embd_out) ||
        !assign_i32("llama.feed_forward_length", model_data.params.n_ff) ||
        !assign_i32("llama.attention.head_count", model_data.params.n_head) ||
        !assign_i32("llama.attention.head_count_kv", model_data.params.n_head_kv) ||
        !assign_i32("llama.rope.dimension_count", model_data.params.n_rot) ||
        !assign_i32("llama.block_count", model_data.params.n_layer) ||
        !assign_i32("llama.vocab_size", model_data.params.n_vocab) ||
        !assign_f32("llama.attention.layer_norm_epsilon",
                    model_data.params.attention_layer_norm_epsilon) ||
        !assign_f32("llama.attention.layer_norm_rms_epsilon",
                    model_data.params.attention_layer_norm_rms_epsilon) ||
        !assign_f32("llama.attention.clamp_kqv", model_data.params.attention_clamp_kqv) ||
        !assign_f32("llama.attn_logit_softcapping", model_data.params.attn_logit_softcapping) ||
        !assign_f32("llama.final_logit_softcapping",
                    model_data.params.final_logit_softcapping) ||
        !assign_f32("llama.residual_scale", model_data.params.residual_scale) ||
        !assign_f32("llama.embedding_scale", model_data.params.embedding_scale) ||
        !assign_f32("llama.rope.freq_base", model_data.params.rope_freq_base) ||
        !assign_f32("llama.rope.freq_base_swa", model_data.params.rope_freq_base_swa) ||
        !assign_bool("llama.use_parallel_residual",
                     model_data.params.use_parallel_residual)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  if (is_qwen3) {
    int32_t qwen3_key_length = 0;
    int32_t qwen3_value_length = 0;
    if (!assign_i32("qwen3.context_length", model_data.params.n_ctx) ||
        !assign_i32("qwen3.embedding_length", model_data.params.n_embd) ||
        !assign_i32("qwen3.feed_forward_length", model_data.params.n_ff) ||
        !assign_i32("qwen3.attention.head_count", model_data.params.n_head) ||
        !assign_i32("qwen3.attention.head_count_kv", model_data.params.n_head_kv) ||
        !assign_i32("qwen3.attention.key_length", qwen3_key_length) ||
        !assign_i32("qwen3.attention.value_length", qwen3_value_length) ||
        !assign_i32("qwen3.block_count", model_data.params.n_layer) ||
        !assign_f32("qwen3.attention.layer_norm_rms_epsilon",
                    model_data.params.attention_layer_norm_rms_epsilon) ||
        !assign_f32("qwen3.rope.freq_base", model_data.params.rope_freq_base)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    model_data.params.attention_key_length = qwen3_key_length;
    model_data.params.attention_value_length = qwen3_value_length;
    if (model_data.params.n_embd_out == 0) {
      model_data.params.n_embd_out = model_data.params.n_embd;
    }
    if (model_data.params.n_rot == 0) {
      model_data.params.n_rot = qwen3_key_length;
    }
    if (qwen3_key_length <= 0 || qwen3_value_length <= 0) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  if (is_lfm2) {
    int32_t lfm2_head_count_kv = 0;
    if (!assign_i32("lfm2.context_length", model_data.params.n_ctx) ||
        !assign_i32("lfm2.embedding_length", model_data.params.n_embd) ||
        !assign_i32("lfm2.feed_forward_length", model_data.params.n_ff) ||
        !assign_i32("lfm2.attention.head_count", model_data.params.n_head) ||
        !assign_i32("lfm2.block_count", model_data.params.n_layer) ||
        !assign_i32("lfm2.vocab_size", model_data.params.n_vocab) ||
        !assign_i32("lfm2.shortconv.l_cache", model_data.params.shortconv_l_cache) ||
        !assign_f32(
            "lfm2.attention.layer_norm_rms_epsilon",
            model_data.params.attention_layer_norm_rms_epsilon) ||
        !assign_f32("lfm2.rope.freq_base", model_data.params.rope_freq_base)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    const auto * head_count_kv_entry = find_kv_entry(fixture, "lfm2.attention.head_count_kv");
    if (head_count_kv_entry == nullptr ||
        !decode_integer_array_first_nonzero(fixture, *head_count_kv_entry, lfm2_head_count_kv)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    model_data.params.n_head_kv = lfm2_head_count_kv;
    if (model_data.params.n_embd_out == 0) {
      model_data.params.n_embd_out = model_data.params.n_embd;
    }
  }

  const auto * tokens_entry = find_kv_entry(fixture, "tokenizer.tokens");
  if (tokens_entry != nullptr) {
    uint32_t token_count = 0u;
    if (!decode_string_array_count(fixture, *tokens_entry, token_count) ||
        token_count > static_cast<uint32_t>(emel::model::data::k_max_vocab_tokens)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    model_data.vocab_data.n_tokens = token_count;
    if (model_data.params.n_vocab == 0) {
      model_data.params.n_vocab = static_cast<int32_t>(token_count);
    }
  }

  return emel::error::cast(emel::model::loader::error::none);
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
  g_generation_architecture_contract.clear();
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
    g_generation_formatter_contract.assign(fixture.formatter_binding.contract);
    return false;
  }
  fixture.formatter_binding = resolve_fixture_formatter_binding(fixture);
  g_generation_architecture_contract.assign(
      emel::model::architecture_name_view(fixture.model_data));
  g_generation_formatter_contract.assign(fixture.formatter_binding.contract);
  if (!emel::tools::generation_formatter_contract::binding_supported(
          fixture.formatter_binding)) {
    return false;
  }
  return true;
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

llama_context_ptr make_reference_context(llama_model * model) {
  llama_context_params params = llama_context_default_params();
  params.flash_attn_type = k_reference_flash_attn_type;
  params.n_ctx = 0;
  params.n_batch = 512;
  params.n_ubatch = 512;
  params.n_seq_max = 1;
  params.n_threads = 1;
  params.n_threads_batch = 1;
  params.embeddings = false;
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

bool tokenize_reference_prompt(const emel_fixture & fixture,
                               const generation_case_spec & spec,
                               std::vector<llama_token> & tokens_out) {
  if (fixture.reference_vocab == nullptr) {
    return false;
  }

  std::string formatted_prompt = {};
  if (!emel::tools::generation_formatter_contract::format_single_user_prompt(
          fixture.formatter_binding, spec.prompt, formatted_prompt)) {
    return false;
  }

  int32_t token_capacity =
      std::max<int32_t>(8, static_cast<int32_t>(formatted_prompt.size()) + 8);
  tokens_out.resize(static_cast<size_t>(token_capacity));
  int32_t token_count = llama_tokenize(fixture.reference_vocab,
                                       formatted_prompt.data(),
                                       static_cast<int32_t>(formatted_prompt.size()),
                                       tokens_out.data(),
                                       token_capacity,
                                       false,
                                       false);
  if (token_count < 0) {
    token_capacity = -token_count;
    tokens_out.resize(static_cast<size_t>(token_capacity));
    token_count = llama_tokenize(fixture.reference_vocab,
                                 formatted_prompt.data(),
                                 static_cast<int32_t>(formatted_prompt.size()),
                                 tokens_out.data(),
                                 token_capacity,
                                 false,
                                 false);
  }
  if (token_count <= 0) {
    return false;
  }

  tokens_out.resize(static_cast<size_t>(token_count));
  return true;
}

bool append_reference_piece(const emel_fixture & fixture,
                            const llama_token token,
                            generation_result & result_out) {
  if (fixture.reference_vocab == nullptr || result_out.output_length >= result_out.output.size()) {
    return false;
  }

  if (llama_vocab_is_control(fixture.reference_vocab, token) ||
      llama_vocab_is_eog(fixture.reference_vocab, token)) {
    return true;
  }

  const char * piece = llama_vocab_get_text(fixture.reference_vocab, token);
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

bool run_reference_generate(const emel_fixture & fixture,
                            const generation_case_spec & spec,
                            generation_seam_audit & seam,
                            generation_result & result_out) {
  if (fixture.reference_model == nullptr || fixture.reference_vocab == nullptr ||
      fixture.reference_vocab_size <= 0) {
    return false;
  }

  llama_context_ptr ctx = make_reference_context(fixture.reference_model.get());
  if (ctx == nullptr) {
    return false;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(fixture, spec, prompt_tokens)) {
    return false;
  }

  result_out = {};
  llama_batch prompt_batch =
      llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
  if (run_direct_reference_decode(seam, ctx.get(), prompt_batch) != 0) {
    return false;
  }

  for (int32_t step = 0; step < spec.max_tokens; ++step) {
    float * logits = read_direct_reference_logits(seam, ctx.get());
    if (logits == nullptr) {
      return false;
    }

    const llama_token selected = select_argmax_token_from_logits(logits, fixture.reference_vocab_size);
    result_out.tokens_generated += 1;
    if (!append_reference_piece(fixture, selected, result_out)) {
      return false;
    }
    if (llama_vocab_is_eog(fixture.reference_vocab, selected)) {
      break;
    }

    llama_token next_token = selected;
    llama_batch decode_batch = llama_batch_get_one(&next_token, 1);
    if (run_direct_reference_decode(seam, ctx.get(), decode_batch) != 0) {
      return false;
    }
  }
  return true;
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

bool run_reference_generate_preloaded(const emel_fixture & fixture,
                                      const generation_case_spec & spec,
                                      llama_context * ctx,
                                      const std::vector<llama_token> & prompt_tokens,
                                      generation_seam_audit & seam,
                                      generation_result & result_out) {
  if (fixture.reference_vocab == nullptr ||
      fixture.reference_vocab_size <= 0 ||
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

    const llama_token selected = select_argmax_token_from_logits(logits, fixture.reference_vocab_size);
    result_out.tokens_generated += 1;
    if (!append_reference_piece(fixture, selected, result_out)) {
      return false;
    }
    if (llama_vocab_is_eog(fixture.reference_vocab, selected)) {
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

emel::bench::config generation_case_config(const emel::bench::config & cfg) {
  emel::bench::config case_cfg = cfg;
  case_cfg.iterations = read_env_u64("EMEL_BENCH_GENERATION_ITERS", 1u);
  case_cfg.runs = read_env_size("EMEL_BENCH_GENERATION_RUNS", cfg.runs);
  case_cfg.warmup_iterations = read_env_u64("EMEL_BENCH_GENERATION_WARMUP_ITERS", 0u);
  case_cfg.warmup_runs = read_env_size("EMEL_BENCH_GENERATION_WARMUP_RUNS", 0u);
  return case_cfg;
}

const std::vector<prepared_generation_fixture> & maintained_generation_fixtures() {
  static const std::vector<prepared_generation_fixture> fixtures = [] {
    ensure_llama_backend_ready();

    std::vector<prepared_generation_fixture> prepared_fixtures(k_generation_fixtures.size());
    for (size_t fixture_index = 0u; fixture_index < k_generation_fixtures.size(); ++fixture_index) {
      const generation_fixture_spec & spec = k_generation_fixtures[fixture_index];
      prepared_generation_fixture & prepared_fixture = prepared_fixtures[fixture_index];
      prepared_fixture.spec = &spec;
      g_generation_fixture_rel = spec.fixture->fixture_rel;

      const std::string model_path = resolve_generation_model_path(spec.fixture->fixture_rel);
      if (!prepare_emel_fixture(prepared_fixture.fixture, model_path)) {
        fail_bench_setup("prepare_emel_fixture", model_path.c_str());
      }

      prepared_fixture.fixture.reference_model = load_reference_model(model_path);
      if (prepared_fixture.fixture.reference_model == nullptr) {
        fail_bench_setup("load_reference_model", model_path.c_str());
      }

      prepared_fixture.fixture.reference_vocab =
          llama_model_get_vocab(prepared_fixture.fixture.reference_model.get());
      if (prepared_fixture.fixture.reference_vocab == nullptr) {
        fail_bench_setup("llama_model_get_vocab", model_path.c_str());
      }
      prepared_fixture.fixture.reference_vocab_size =
          llama_vocab_n_tokens(prepared_fixture.fixture.reference_vocab);
      if (!emel::model::detail::load_vocab_from_gguf(
              kv_binding_from_fixture(prepared_fixture.fixture),
              prepared_fixture.fixture.model_data.vocab_data)) {
        fail_bench_setup("load_vocab_from_gguf", model_path.c_str());
      }
      prepared_fixture.fixture.model_data.params.n_vocab =
          static_cast<int32_t>(prepared_fixture.fixture.model_data.vocab_data.n_tokens);
      prepared_fixture.fixture.fallback_token_id =
          find_generation_fallback_token_id(prepared_fixture.fixture.model_data.vocab_data);
    }

    return prepared_fixtures;
  }();

  return fixtures;
}

}  // namespace

namespace emel::bench {

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

void append_emel_generation_cases(std::vector<result> & results, const config & cfg) {
  const auto & fixtures = maintained_generation_fixtures();
  const config case_cfg = generation_case_config(cfg);

  reset_generation_flash_evidence();
  for (const prepared_generation_fixture & prepared_fixture : fixtures) {
    const emel_fixture & fixture = prepared_fixture.fixture;
    g_generation_fixture_rel = prepared_fixture.spec->fixture->fixture_rel;
    for (const generation_case_spec & generation_case : prepared_fixture.spec->cases) {
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
      auto session = std::make_unique<emel_session>();
      prepare_emel_session(fixture, *session);
      g_generation_architecture_contract.assign(
          emel::model::architecture_name_view(session->model_data));
      g_generation_formatter_contract.assign(session->formatter_binding.contract);
      if (!initialize_emel_session(*session, generation_case)) {
        fail_bench_setup("initialize_emel_session", generation_case.name.data());
      }

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
      if (prepared_fixture.spec->fixture->current_publication &&
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
  const auto & fixtures = maintained_generation_fixtures();
  const config case_cfg = generation_case_config(cfg);

  for (const prepared_generation_fixture & prepared_fixture : fixtures) {
    const emel_fixture & fixture = prepared_fixture.fixture;
    g_generation_fixture_rel = prepared_fixture.spec->fixture->fixture_rel;
    for (const generation_case_spec & generation_case : prepared_fixture.spec->cases) {
      volatile std::size_t sink = 0u;
      generation_seam_audit seam = {};
      auto fn = [&]() {
        reset_generation_seam(seam);
        generation_result generated{};
        // Keep the reference path honest with EMEL's timed generate request:
        // formatting and tokenization stay inside the measured lambda here too.
        if (!run_reference_generate(fixture, generation_case, seam, generated)) {
          fail_bench_setup("run_reference_generate", generation_case.name.data());
        }
        sink ^= generated.output_length;
      };

      results.push_back(measure_case(generation_case.name.data(), case_cfg, fn));
      if (generation_seam_audit_enabled()) {
        print_generation_seam_audit("reference", seam);
        verify_reference_generation_seam(seam);
      }
      static_cast<void>(sink);
    }
  }
}

}  // namespace emel::bench
