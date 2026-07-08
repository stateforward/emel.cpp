#include <array>
#include <stateforward/sml.hpp>
#include <cstdint>
#include <cstring>
#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "emel/docs/detail.hpp"
#include "emel/emel.h"
#include "emel/text/generator/errors.hpp"
#include "emel/text/generator/initializer/sm.hpp"
#include "emel/text/generator/prefill/sm.hpp"
#include "emel/text/generator/sm.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/graph/tensor/errors.hpp"
#include "emel/graph/tensor/events.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/sm.hpp"
#include "../../allocation_tracker.hpp"
#include "generator_test_policies.hpp"

namespace {

std::filesystem::path repo_root() {
#ifdef EMEL_TEST_REPO_ROOT
  return std::filesystem::path{EMEL_TEST_REPO_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

constexpr bool host_is_aarch64() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
  return true;
#else
  return false;
#endif
}

constexpr bool host_is_x86_64() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  return true;
#else
  return false;
#endif
}

struct callback_tracker {
  bool initialize_done_called = false;
  bool initialize_error_called = false;
  bool generate_done_called = false;
  bool generate_error_called = false;
  const emel::text::generator::event::initialize * initialize_request = nullptr;
  const emel::text::generator::event::generate * generate_request = nullptr;
  int32_t tokens_generated = -1;
  size_t output_length = 0;
  emel::error::type err = emel::error::cast(emel::text::generator::error::none);
};

void on_initialize_done(void * owner, const emel::text::generator::events::initialize_done & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->initialize_done_called = true;
  tracker->initialize_request = ev.request;
}

void on_initialize_error(void * owner, const emel::text::generator::events::initialize_error & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->initialize_error_called = true;
  tracker->initialize_request = ev.request;
  tracker->err = ev.err;
}

void on_generate_done(void * owner, const emel::text::generator::events::generation_done & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->generate_done_called = true;
  tracker->generate_request = ev.request;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->output_length = ev.output_length;
}

void on_generate_error(void * owner, const emel::text::generator::events::generation_error & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->generate_error_called = true;
  tracker->generate_request = ev.request;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->output_length = ev.output_length;
  tracker->err = ev.err;
}

int32_t add_token(emel::model::data::vocab & vocab, const char * text,
                  const int32_t type = 0) {
  const uint32_t length = static_cast<uint32_t>(std::strlen(text));
  const uint32_t offset = vocab.token_bytes_used;
  std::memcpy(vocab.token_storage.data() + offset, text, length);
  const uint32_t id = vocab.n_tokens;
  vocab.entries[id].text_offset = offset;
  vocab.entries[id].text_length = length;
  vocab.entries[id].score = 0.0f;
  vocab.entries[id].type = type;
  vocab.token_bytes_used += length;
  vocab.n_tokens = id + 1;
  return static_cast<int32_t>(id);
}

bool tokenizer_bind_dispatch(void * tokenizer_sm,
                             const emel::text::tokenizer::event::bind & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

bool tokenizer_tokenize_dispatch(
    void * tokenizer_sm, const emel::text::tokenizer::event::tokenize & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
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

template <class... Ts, class fn>
constexpr void for_each_type(stateforward::sml::aux::type_list<Ts...>, fn && visitor) {
  (visitor.template operator()<Ts>(), ...);
}

struct prepared_model {
  emel::model::data data = {};
  std::vector<std::vector<float>> tensor_storage = {};
  int32_t hello_id = -1;
  int32_t world_id = -1;
};

static_assert(std::is_reference_v<
              decltype(std::declval<const emel::text::generator::event::initialize &>()
                           .dispatch_tokenizer_bind)>);
static_assert(std::is_reference_v<
              decltype(std::declval<const emel::text::generator::event::initialize &>()
                           .dispatch_tokenizer_tokenize)>);
static_assert(std::is_same_v<
              std::remove_cvref_t<
                  decltype(std::declval<const emel::text::generator::event::initialize &>()
                               .sampler_fns)>,
              std::span<emel::logits::sampler::fn>>);
static_assert(std::is_same_v<
              std::remove_cvref_t<
                  decltype(std::declval<const emel::text::generator::event::generate &>().messages)>,
              std::span<const emel::text::formatter::chat_message>>);
static_assert(std::is_same_v<
              std::remove_cvref_t<
                  decltype(std::declval<const emel::text::generator::event::generate &>().output)>,
              std::span<char>>);
static_assert(std::is_reference_v<
              decltype(std::declval<const emel::text::generator::event::generate &>()
                           .output_length_out)>);

emel::text::generator::diagnostics capture_generator_diagnostics(
    emel::text::generator::sm & generator) {
  emel::text::generator::diagnostics diagnostics = {};
  CHECK(generator.process_event(emel::text::generator::event::capture_diagnostics{diagnostics}));
  return diagnostics;
}

struct checked_formatter_ctx {
  std::span<const emel::text::formatter::chat_message> expected_messages = {};
  bool expected_add_generation_prompt = false;
  bool expected_enable_thinking = false;
  std::string_view formatted = {};
  bool seen = false;
};

bool format_checked_messages(void * formatter_ctx,
                             const emel::text::formatter::format_request & request,
                             int32_t * error_out) {
  if (error_out != nullptr) {
    *error_out = emel::text::formatter::error_code(emel::text::formatter::error::none);
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0u;
  }
  if (formatter_ctx == nullptr || request.output == nullptr ||
      request.output_capacity < request.messages.size()) {
    if (error_out != nullptr) {
      *error_out = emel::text::formatter::error_code(
          emel::text::formatter::error::invalid_request);
    }
    return false;
  }

  auto & ctx = *static_cast<checked_formatter_ctx *>(formatter_ctx);
  if (request.messages.size() != ctx.expected_messages.size() ||
      request.add_generation_prompt != ctx.expected_add_generation_prompt ||
      request.enable_thinking != ctx.expected_enable_thinking) {
    if (error_out != nullptr) {
      *error_out = emel::text::formatter::error_code(
          emel::text::formatter::error::invalid_request);
    }
    return false;
  }

  for (size_t idx = 0; idx < request.messages.size(); ++idx) {
    if (request.messages[idx].role != ctx.expected_messages[idx].role ||
        request.messages[idx].content != ctx.expected_messages[idx].content) {
      if (error_out != nullptr) {
        *error_out = emel::text::formatter::error_code(
            emel::text::formatter::error::invalid_request);
      }
      return false;
    }
  }

  if (request.output_capacity < ctx.formatted.size()) {
    if (error_out != nullptr) {
      *error_out = emel::text::formatter::error_code(
          emel::text::formatter::error::invalid_request);
    }
    return false;
  }

  if (!ctx.formatted.empty()) {
    std::memcpy(request.output, ctx.formatted.data(), ctx.formatted.size());
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = ctx.formatted.size();
  }
  ctx.seen = true;
  return true;
}

void build_prepared_model(prepared_model & prepared) {
  prepared.tensor_storage.reserve(12);
  prepared.data.vocab_data.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  prepared.data.vocab_data.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  prepared.data.vocab_data.ignore_merges = true;
  prepared.hello_id = add_token(prepared.data.vocab_data, "hello");
  prepared.world_id = add_token(prepared.data.vocab_data, "world");
  prepared.data.params.n_vocab = static_cast<int32_t>(prepared.data.vocab_data.n_tokens);
  prepared.data.params.n_embd = 4;
  prepared.data.params.n_head = 1;
  prepared.data.params.n_head_kv = 1;
  prepared.data.params.n_ctx = 8;
  prepared.data.params.n_rot = 2;
  prepared.data.params.n_layer = 1;
  prepared.data.n_layers = 1;
  prepared.data.weights_data = prepared.data.tensors.data();
  prepared.data.weights_size = 1u;
  std::memcpy(prepared.data.architecture_name.data(), "llama", 5u);

  uint32_t tensor_index = 0u;
  const auto add_name = [&](emel::model::data::tensor_record & tensor, const std::string_view name) {
    tensor.name_offset = prepared.data.name_bytes_used;
    tensor.name_length = static_cast<uint32_t>(name.size());
    std::memcpy(prepared.data.name_storage.data() + prepared.data.name_bytes_used,
                name.data(),
                name.size());
    prepared.data.name_bytes_used += static_cast<uint32_t>(name.size());
  };
  const auto add_vector = [&](const std::string_view name, const std::vector<float> & values) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.push_back(values);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 1;
    tensor.dims[0] = static_cast<int64_t>(values.size());
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
  };
  const auto add_matrix = [&](const std::string_view name,
                              const int32_t rows,
                              const int32_t cols,
                              const std::vector<float> & values) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.push_back(values);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 2;
    tensor.dims[0] = cols;
    tensor.dims[1] = rows;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
  };

  add_matrix("token_embd.weight", 2, 4, {1.0f, 0.0f, 0.0f, 0.0f,
                                         1.0f, 0.0f, 0.0f, 0.0f});
  add_vector("output_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  add_matrix("output.weight", 2, 4, {0.0f, 0.0f, 0.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 0.0f});
  add_vector("blk.0.attn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  add_matrix("blk.0.attn_q.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.attn_k.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.attn_v.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.attn_output.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_vector("blk.0.ffn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  add_matrix("blk.0.ffn_gate.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.ffn_down.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.ffn_up.weight", 4, 4, std::vector<float>(16, 0.0f));
  prepared.data.n_tensors = tensor_index;
}

void build_quantized_contract_prepared_model(prepared_model & prepared) {
  constexpr int32_t k_n_embd =
      static_cast<int32_t>(emel::kernel::detail::quant::QK_K);
  prepared.tensor_storage.reserve(12);
  prepared.data.vocab_data.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  prepared.data.vocab_data.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  prepared.data.vocab_data.ignore_merges = true;
  prepared.hello_id = add_token(prepared.data.vocab_data, "hello");
  prepared.world_id = add_token(prepared.data.vocab_data, "world");
  prepared.data.params.n_vocab = static_cast<int32_t>(prepared.data.vocab_data.n_tokens);
  prepared.data.params.n_embd = k_n_embd;
  prepared.data.params.n_head = 1;
  prepared.data.params.n_head_kv = 1;
  prepared.data.params.n_ctx = 8;
  prepared.data.params.n_rot = 2;
  prepared.data.params.attention_key_length = k_n_embd;
  prepared.data.params.attention_value_length = k_n_embd;
  prepared.data.params.n_layer = 1;
  prepared.data.n_layers = 1;
  prepared.data.weights_data = prepared.data.tensors.data();
  prepared.data.weights_size = 1u;
  std::memcpy(prepared.data.architecture_name.data(), "llama", 5u);

  uint32_t tensor_index = 0u;
  const auto add_name = [&](emel::model::data::tensor_record & tensor, const std::string_view name) {
    tensor.name_offset = prepared.data.name_bytes_used;
    tensor.name_length = static_cast<uint32_t>(name.size());
    std::memcpy(prepared.data.name_storage.data() + prepared.data.name_bytes_used,
                name.data(),
                name.size());
    prepared.data.name_bytes_used += static_cast<uint32_t>(name.size());
  };
  const auto add_vector = [&](const std::string_view name, const int32_t cols) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.emplace_back(static_cast<size_t>(cols), 1.0f);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 1;
    tensor.dims[0] = cols;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(prepared.tensor_storage.back().size() * sizeof(float));
  };
  const auto add_matrix = [&](const std::string_view name, const int32_t rows, const int32_t cols) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.emplace_back(static_cast<size_t>(rows) * static_cast<size_t>(cols),
                                         0.0f);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 2;
    tensor.dims[0] = cols;
    tensor.dims[1] = rows;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(prepared.tensor_storage.back().size() * sizeof(float));
  };

  add_matrix("token_embd.weight", 2, k_n_embd);
  add_vector("output_norm.weight", k_n_embd);
  add_matrix("output.weight", 2, k_n_embd);
  add_vector("blk.0.attn_norm.weight", k_n_embd);
  add_matrix("blk.0.attn_q.weight", k_n_embd, k_n_embd);
  add_matrix("blk.0.attn_k.weight", k_n_embd, k_n_embd);
  add_matrix("blk.0.attn_v.weight", k_n_embd, k_n_embd);
  add_matrix("blk.0.attn_output.weight", k_n_embd, k_n_embd);
  add_vector("blk.0.ffn_norm.weight", k_n_embd);
  add_matrix("blk.0.ffn_gate.weight", k_n_embd, k_n_embd);
  add_matrix("blk.0.ffn_down.weight", k_n_embd, k_n_embd);
  add_matrix("blk.0.ffn_up.weight", k_n_embd, k_n_embd);
  prepared.data.n_tensors = tensor_index;
}

void build_qwen3_quantized_contract_prepared_model(prepared_model & prepared,
                                                   const bool include_q_norm,
                                                   const bool include_k_norm) {
  constexpr int32_t k_n_embd =
      static_cast<int32_t>(emel::kernel::detail::quant::QK_K);
  prepared.tensor_storage.reserve(14);
  prepared.data.vocab_data.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  prepared.data.vocab_data.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  prepared.data.vocab_data.ignore_merges = true;
  prepared.hello_id = add_token(prepared.data.vocab_data, "hello");
  prepared.world_id = add_token(prepared.data.vocab_data, "world");
  prepared.data.params.n_vocab = static_cast<int32_t>(prepared.data.vocab_data.n_tokens);
  prepared.data.params.n_embd = k_n_embd;
  prepared.data.params.n_head = 1;
  prepared.data.params.n_head_kv = 1;
  prepared.data.params.n_ctx = 8;
  prepared.data.params.n_rot = 2;
  prepared.data.params.n_layer = 1;
  prepared.data.n_layers = 1;
  prepared.data.weights_data = prepared.data.tensors.data();
  prepared.data.weights_size = 1u;
  std::memcpy(prepared.data.architecture_name.data(), "qwen3", 5u);

  uint32_t tensor_index = 0u;
  const auto add_name = [&](emel::model::data::tensor_record & tensor, const std::string_view name) {
    tensor.name_offset = prepared.data.name_bytes_used;
    tensor.name_length = static_cast<uint32_t>(name.size());
    std::memcpy(prepared.data.name_storage.data() + prepared.data.name_bytes_used,
                name.data(),
                name.size());
    prepared.data.name_bytes_used += static_cast<uint32_t>(name.size());
  };
  const auto add_vector = [&](const std::string_view name, const int32_t cols) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.emplace_back(static_cast<size_t>(cols), 1.0f);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 1;
    tensor.dims[0] = cols;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(prepared.tensor_storage.back().size() * sizeof(float));
  };
  const auto add_matrix = [&](const std::string_view name, const int32_t rows, const int32_t cols) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.emplace_back(static_cast<size_t>(rows) * static_cast<size_t>(cols),
                                         0.0f);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 2;
    tensor.dims[0] = cols;
    tensor.dims[1] = rows;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(prepared.tensor_storage.back().size() * sizeof(float));
  };

  add_matrix("token_embd.weight", 2, k_n_embd);
  add_vector("output_norm.weight", k_n_embd);
  add_matrix("output.weight", 2, k_n_embd);
  add_vector("blk.0.attn_norm.weight", k_n_embd);
  add_matrix("blk.0.attn_q.weight", k_n_embd, k_n_embd);
  add_matrix("blk.0.attn_k.weight", k_n_embd, k_n_embd);
  add_matrix("blk.0.attn_v.weight", k_n_embd, k_n_embd);
  if (include_q_norm) {
    add_vector("blk.0.attn_q_norm.weight", k_n_embd);
  }
  if (include_k_norm) {
    add_vector("blk.0.attn_k_norm.weight", k_n_embd);
  }
  add_matrix("blk.0.attn_output.weight", k_n_embd, k_n_embd);
  add_vector("blk.0.ffn_norm.weight", k_n_embd);
  add_matrix("blk.0.ffn_gate.weight", k_n_embd, k_n_embd);
  add_matrix("blk.0.ffn_down.weight", k_n_embd, k_n_embd);
  add_matrix("blk.0.ffn_up.weight", k_n_embd, k_n_embd);
  prepared.data.n_tensors = tensor_index;
}

void build_lfm2_quantized_contract_prepared_model(prepared_model & prepared) {
  constexpr int32_t k_n_embd =
      static_cast<int32_t>(emel::kernel::detail::quant::QK_K);
  prepared.tensor_storage.reserve(30);
  prepared.data.vocab_data.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  prepared.data.vocab_data.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  prepared.data.vocab_data.ignore_merges = true;
  prepared.hello_id = add_token(prepared.data.vocab_data, "hello");
  prepared.world_id = add_token(prepared.data.vocab_data, "world");
  prepared.data.params.n_vocab = static_cast<int32_t>(prepared.data.vocab_data.n_tokens);
  prepared.data.params.n_embd = k_n_embd;
  prepared.data.params.n_head = 1;
  prepared.data.params.n_head_kv = 1;
  prepared.data.params.n_ctx = 8;
  prepared.data.params.n_rot = 2;
  prepared.data.params.n_layer = 3;
  prepared.data.n_layers = 3;
  prepared.data.weights_data = prepared.data.tensors.data();
  prepared.data.weights_size = 1u;
  std::memcpy(prepared.data.architecture_name.data(), "lfm2", 4u);

  uint32_t tensor_index = 0u;
  const auto add_name = [&](emel::model::data::tensor_record & tensor, const std::string_view name) {
    tensor.name_offset = prepared.data.name_bytes_used;
    tensor.name_length = static_cast<uint32_t>(name.size());
    std::memcpy(prepared.data.name_storage.data() + prepared.data.name_bytes_used,
                name.data(),
                name.size());
    prepared.data.name_bytes_used += static_cast<uint32_t>(name.size());
  };
  const auto add_vector = [&](const std::string_view name, const int32_t cols) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.emplace_back(static_cast<size_t>(cols), 1.0f);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 1;
    tensor.dims[0] = cols;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(prepared.tensor_storage.back().size() * sizeof(float));
  };
  const auto add_matrix = [&](const std::string_view name, const int32_t rows, const int32_t cols) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.emplace_back(static_cast<size_t>(rows) * static_cast<size_t>(cols),
                                         0.0f);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 2;
    tensor.dims[0] = cols;
    tensor.dims[1] = rows;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(prepared.tensor_storage.back().size() * sizeof(float));
  };
  const auto add_common_block = [&](const int32_t block_index) {
    const std::string prefix = "blk." + std::to_string(block_index) + ".";
    add_vector(prefix + "attn_norm.weight", k_n_embd);
    add_vector(prefix + "ffn_norm.weight", k_n_embd);
    add_matrix(prefix + "ffn_gate.weight", k_n_embd, k_n_embd);
    add_matrix(prefix + "ffn_down.weight", k_n_embd, k_n_embd);
    add_matrix(prefix + "ffn_up.weight", k_n_embd, k_n_embd);
  };

  add_matrix("token_embd.weight", 2, k_n_embd);
  add_vector("token_embd_norm.weight", k_n_embd);
  add_common_block(0);
  add_matrix("blk.0.shortconv.conv.weight", 3, k_n_embd);
  add_matrix("blk.0.shortconv.in_proj.weight", 3 * k_n_embd, k_n_embd);
  add_matrix("blk.0.shortconv.out_proj.weight", k_n_embd, k_n_embd);
  add_common_block(1);
  add_matrix("blk.1.shortconv.conv.weight", 3, k_n_embd);
  add_matrix("blk.1.shortconv.in_proj.weight", 3 * k_n_embd, k_n_embd);
  add_matrix("blk.1.shortconv.out_proj.weight", k_n_embd, k_n_embd);
  add_common_block(2);
  add_matrix("blk.2.attn_q.weight", k_n_embd, k_n_embd);
  add_matrix("blk.2.attn_k.weight", k_n_embd, k_n_embd);
  add_matrix("blk.2.attn_v.weight", k_n_embd, k_n_embd);
  add_vector("blk.2.attn_q_norm.weight", k_n_embd);
  add_vector("blk.2.attn_k_norm.weight", k_n_embd);
  add_matrix("blk.2.attn_output.weight", k_n_embd, k_n_embd);
  prepared.data.n_tensors = tensor_index;
}

void build_qwen3_prepared_model(prepared_model & prepared) {
  prepared.tensor_storage.reserve(14);
  prepared.data.vocab_data.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  prepared.data.vocab_data.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  prepared.data.vocab_data.ignore_merges = true;
  prepared.hello_id = add_token(prepared.data.vocab_data, "hello");
  prepared.world_id = add_token(prepared.data.vocab_data, "world");
  prepared.data.params.n_vocab = static_cast<int32_t>(prepared.data.vocab_data.n_tokens);
  prepared.data.params.n_embd = 4;
  prepared.data.params.n_head = 2;
  prepared.data.params.n_head_kv = 2;
  prepared.data.params.n_ctx = 8;
  prepared.data.params.n_rot = 2;
  prepared.data.params.attention_key_length = 2;
  prepared.data.params.attention_value_length = 2;
  prepared.data.params.n_layer = 1;
  prepared.data.n_layers = 1;
  prepared.data.weights_data = prepared.data.tensors.data();
  prepared.data.weights_size = 1u;
  std::memcpy(prepared.data.architecture_name.data(), "qwen3", 5u);

  uint32_t tensor_index = 0u;
  const auto add_name = [&](emel::model::data::tensor_record & tensor, const std::string_view name) {
    tensor.name_offset = prepared.data.name_bytes_used;
    tensor.name_length = static_cast<uint32_t>(name.size());
    std::memcpy(prepared.data.name_storage.data() + prepared.data.name_bytes_used,
                name.data(),
                name.size());
    prepared.data.name_bytes_used += static_cast<uint32_t>(name.size());
  };
  const auto add_vector = [&](const std::string_view name, const std::vector<float> & values) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.push_back(values);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 1;
    tensor.dims[0] = static_cast<int64_t>(values.size());
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
  };
  const auto add_matrix = [&](const std::string_view name,
                              const int32_t rows,
                              const int32_t cols,
                              const std::vector<float> & values) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.push_back(values);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 2;
    tensor.dims[0] = cols;
    tensor.dims[1] = rows;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
  };

  add_matrix("token_embd.weight", 2, 4, {1.0f, 0.0f, 0.0f, 0.0f,
                                         1.0f, 0.0f, 0.0f, 0.0f});
  add_vector("output_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  add_matrix("output.weight", 2, 4, {0.0f, 0.0f, 0.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 0.0f});
  add_vector("blk.0.attn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  add_matrix("blk.0.attn_q.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.attn_k.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.attn_v.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_vector("blk.0.attn_q_norm.weight", {1.0f, 1.0f});
  add_vector("blk.0.attn_k_norm.weight", {1.0f, 1.0f});
  add_matrix("blk.0.attn_output.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_vector("blk.0.ffn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  add_matrix("blk.0.ffn_gate.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.ffn_down.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.ffn_up.weight", 4, 4, std::vector<float>(16, 0.0f));
  prepared.data.n_tensors = tensor_index;
}

void build_gemma4_prepared_model(prepared_model & prepared) {
  prepared.tensor_storage.reserve(13);
  prepared.data.vocab_data.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  prepared.data.vocab_data.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  prepared.data.vocab_data.ignore_merges = true;
  prepared.hello_id = add_token(prepared.data.vocab_data, "hello");
  prepared.world_id = add_token(prepared.data.vocab_data, "world");
  prepared.data.params.n_vocab = static_cast<int32_t>(prepared.data.vocab_data.n_tokens);
  prepared.data.params.n_embd = 4;
  prepared.data.params.n_embd_out = 4;
  prepared.data.params.n_ff = 4;
  prepared.data.params.n_head = 2;
  prepared.data.params.n_head_kv = 1;
  prepared.data.params.n_ctx = 8;
  prepared.data.params.n_rot = 2;
  prepared.data.params.attention_key_length = 2;
  prepared.data.params.attention_value_length = 2;
  prepared.data.params.n_layer = 1;
  prepared.data.params.attention_layer_norm_rms_epsilon = 1.0e-6f;
  prepared.data.params.rope_freq_base = 10000.0f;
  prepared.data.params.tie_word_embeddings = true;
  prepared.data.n_layers = 1;
  prepared.data.weights_data = prepared.data.tensors.data();
  prepared.data.weights_size = 1u;
  std::memcpy(prepared.data.architecture_name.data(), "gemma4", 6u);

  uint32_t tensor_index = 0u;
  const auto add_name = [&](emel::model::data::tensor_record & tensor, const std::string_view name) {
    tensor.name_offset = prepared.data.name_bytes_used;
    tensor.name_length = static_cast<uint32_t>(name.size());
    std::memcpy(prepared.data.name_storage.data() + prepared.data.name_bytes_used,
                name.data(),
                name.size());
    prepared.data.name_bytes_used += static_cast<uint32_t>(name.size());
  };
  const auto add_vector = [&](const std::string_view name, const std::vector<float> & values) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.push_back(values);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 1;
    tensor.dims[0] = static_cast<int64_t>(values.size());
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
  };
  const auto add_matrix = [&](const std::string_view name,
                              const int32_t rows,
                              const int32_t cols,
                              const std::vector<float> & values) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.push_back(values);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 2;
    tensor.dims[0] = cols;
    tensor.dims[1] = rows;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
  };

  add_matrix("token_embd.weight", 2, 4, {1.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 1.0f, 0.0f, 0.0f});
  add_vector("output_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  add_vector("blk.0.attn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  add_matrix("blk.0.attn_q.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.attn_k.weight", 2, 4, std::vector<float>(8, 0.0f));
  add_matrix("blk.0.attn_v.weight", 2, 4, std::vector<float>(8, 0.0f));
  add_vector("blk.0.attn_q_norm.weight", {1.0f, 1.0f});
  add_vector("blk.0.attn_k_norm.weight", {1.0f, 1.0f});
  add_matrix("blk.0.attn_output.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_vector("blk.0.ffn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  add_matrix("blk.0.ffn_gate.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.ffn_down.weight", 4, 4, std::vector<float>(16, 0.0f));
  add_matrix("blk.0.ffn_up.weight", 4, 4, std::vector<float>(16, 0.0f));
  prepared.data.n_tensors = tensor_index;
}

void build_gemma4_multiwidth_prepared_model(prepared_model & prepared) {
  prepared.tensor_storage.reserve(5 * 13);
  prepared.data.vocab_data.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  prepared.data.vocab_data.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  prepared.data.vocab_data.ignore_merges = true;
  prepared.hello_id = add_token(prepared.data.vocab_data, "hello");
  prepared.world_id = add_token(prepared.data.vocab_data, "world");
  prepared.data.params.n_vocab = static_cast<int32_t>(prepared.data.vocab_data.n_tokens);
  prepared.data.params.n_embd = 4;
  prepared.data.params.n_embd_out = 4;
  prepared.data.params.n_ff = 4;
  prepared.data.params.n_head = 2;
  prepared.data.params.n_head_kv = 1;
  prepared.data.params.n_ctx = 8;
  prepared.data.params.n_rot = 4;
  prepared.data.params.n_rot_swa = 2;
  prepared.data.params.attention_key_length = 4;
  prepared.data.params.attention_key_length_swa = 2;
  prepared.data.params.attention_value_length = 4;
  prepared.data.params.attention_value_length_swa = 2;
  prepared.data.params.n_layer = 5;
  prepared.data.params.full_attention_interval = 5;
  prepared.data.params.embd_length_per_layer_input = 2;
  prepared.data.params.attention_sliding_window = 4;
  prepared.data.params.attention_shared_kv_layers = 0;
  prepared.data.params.attention_layer_norm_rms_epsilon = 1.0e-6f;
  prepared.data.params.rope_freq_base = 1000000.0f;
  prepared.data.params.rope_freq_base_swa = 10000.0f;
  prepared.data.params.tie_word_embeddings = true;
  prepared.data.params.attention_sliding_window_pattern_count = 5u;
  prepared.data.params.attention_sliding_window_pattern_flags = {1u, 1u, 1u, 1u, 0u};
  prepared.data.n_layers = 5;
  prepared.data.weights_data = prepared.data.tensors.data();
  prepared.data.weights_size = 1u;
  std::memcpy(prepared.data.architecture_name.data(), "gemma4", 6u);

  uint32_t tensor_index = 0u;
  const auto add_name = [&](emel::model::data::tensor_record & tensor, const std::string_view name) {
    tensor.name_offset = prepared.data.name_bytes_used;
    tensor.name_length = static_cast<uint32_t>(name.size());
    std::memcpy(prepared.data.name_storage.data() + prepared.data.name_bytes_used,
                name.data(),
                name.size());
    prepared.data.name_bytes_used += static_cast<uint32_t>(name.size());
  };
  const auto add_vector = [&](const std::string_view name, const std::vector<float> & values) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.push_back(values);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 1;
    tensor.dims[0] = static_cast<int64_t>(values.size());
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
  };
  const auto add_matrix = [&](const std::string_view name,
                              const int32_t rows,
                              const int32_t cols,
                              const std::vector<float> & values) {
    auto & tensor = prepared.data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.push_back(values);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 2;
    tensor.dims[0] = cols;
    tensor.dims[1] = rows;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
  };
  const auto add_block = [&](const int32_t block, const int32_t q_rows, const int32_t kv_rows) {
    const std::string prefix = "blk." + std::to_string(block) + ".";
    add_vector(prefix + "attn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
    add_matrix(prefix + "attn_q.weight", q_rows, 4, std::vector<float>(q_rows * 4, 0.0f));
    add_matrix(prefix + "attn_k.weight", kv_rows, 4, std::vector<float>(kv_rows * 4, 0.0f));
    add_matrix(prefix + "attn_v.weight", kv_rows, 4, std::vector<float>(kv_rows * 4, 0.0f));
    add_vector(prefix + "attn_q_norm.weight", std::vector<float>(static_cast<size_t>(q_rows / 2), 1.0f));
    add_vector(prefix + "attn_k_norm.weight", std::vector<float>(static_cast<size_t>(kv_rows), 1.0f));
    add_matrix(prefix + "attn_output.weight", 4, q_rows, std::vector<float>(q_rows * 4, 0.0f));
    add_vector(prefix + "ffn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
    add_matrix(prefix + "ffn_gate.weight", 4, 4, std::vector<float>(16, 0.0f));
    add_matrix(prefix + "ffn_down.weight", 4, 4, std::vector<float>(16, 0.0f));
    add_matrix(prefix + "ffn_up.weight", 4, 4, std::vector<float>(16, 0.0f));
  };

  add_matrix("token_embd.weight", 2, 4, {1.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 1.0f, 0.0f, 0.0f});
  add_vector("output_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  for (int32_t block = 0; block < 4; ++block) {
    add_block(block, 4, 2);
  }
  add_block(4, 8, 4);
  prepared.data.n_tensors = tensor_index;
}

emel::model::data & stabilize_model(prepared_model & prepared) {
  prepared.data.weights_data = prepared.data.tensors.data();
  return prepared.data;
}

emel::model::data::tensor_record * find_tensor(prepared_model & prepared,
                                               const std::string_view name) {
  for (uint32_t idx = 0; idx < prepared.data.n_tensors; ++idx) {
    auto & tensor = prepared.data.tensors[idx];
    if (emel::model::tensor_name_view(prepared.data, tensor) == name) {
      return &tensor;
    }
  }
  return nullptr;
}

const emel::model::llama::detail::quantized_stage_audit & find_stage_audit(
    const emel::model::llama::detail::quantized_path_audit & audit,
    const emel::model::llama::detail::quantized_stage_family family) {
  for (const auto & stage : audit.stages) {
    if (stage.family == family) {
      return stage;
    }
  }
  return audit.stages[0];
}

void apply_flash_kv_width_mismatch(prepared_model & prepared) {
  const auto tensor_name = [&](const emel::model::data::tensor_record & tensor) {
    return std::string_view(prepared.data.name_storage.data() + tensor.name_offset,
                            tensor.name_length);
  };

  for (uint32_t idx = 0; idx < prepared.data.n_tensors; ++idx) {
    auto & tensor = prepared.data.tensors[idx];
    const std::string_view name = tensor_name(tensor);
    if (name == "blk.0.attn_k.weight" || name == "blk.0.attn_v.weight") {
      tensor.dims[1] = 2;
      tensor.data_size = static_cast<uint64_t>(2 * 4 * sizeof(float));
    }
  }
}

void apply_quantized_contract_tensor_types(prepared_model & prepared) {
  find_tensor(prepared, "token_embd.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q2_k);
  find_tensor(prepared, "output.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q6_k);
  find_tensor(prepared, "blk.0.attn_q.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q2_k);
  find_tensor(prepared, "blk.0.attn_k.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q2_k);
  find_tensor(prepared, "blk.0.attn_v.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q3_k);
  find_tensor(prepared, "blk.0.attn_output.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q3_k);
  find_tensor(prepared, "blk.0.ffn_gate.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q2_k);
  find_tensor(prepared, "blk.0.ffn_down.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q3_k);
  find_tensor(prepared, "blk.0.ffn_up.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q2_k);
}

void apply_qwen3_quantized_contract_tensor_types(prepared_model & prepared) {
  apply_quantized_contract_tensor_types(prepared);
  find_tensor(prepared, "blk.0.attn_q_norm.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::f32);
  find_tensor(prepared, "blk.0.attn_k_norm.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::f32);
}

struct generator_fixture {
  static constexpr std::string_view k_phase_4_user_content = "hello";
  static constexpr std::array<emel::text::formatter::chat_message, 1> k_phase_4_messages = {
      emel::text::formatter::chat_message{
          .role = "user",
          .content = k_phase_4_user_content,
      },
  };
  static constexpr bool k_phase_4_add_generation_prompt = false;
  static constexpr bool k_phase_4_enable_thinking = false;
  static constexpr int32_t k_phase_4_max_tokens = 1;

  prepared_model prepared = {};
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  emel::text::generator::matmul::lane_pool<7u, 128u, 1048576u> parallel_matmul_lanes = {};
  std::unique_ptr<emel::text::generator::sm> generator = {};
  std::array<emel::logits::sampler::fn, 1> samplers = {
      emel::logits::sampler::fn::from<sampler_select_argmax>(),
  };
  int32_t hello_id = -1;
  int32_t world_id = -1;

  enum class model_variant {
    canonical,
    flash_kv_width_mismatch,
    quantized_contract,
    qwen3_canonical,
    gemma4_canonical,
    gemma4_multiwidth,
  };

  explicit generator_fixture(
      const model_variant variant = model_variant::canonical,
      void * formatter_ctx = nullptr,
      emel::text::formatter::format_fn format_prompt =
          emel::text::formatter::format_raw,
      const emel::memory::hybrid::kv_binding & kv_cache = {})
      : prepared() {
    if (variant == model_variant::quantized_contract) {
      build_quantized_contract_prepared_model(prepared);
      apply_quantized_contract_tensor_types(prepared);
    } else if (variant == model_variant::qwen3_canonical) {
      build_qwen3_prepared_model(prepared);
    } else if (variant == model_variant::gemma4_canonical) {
      build_gemma4_prepared_model(prepared);
    } else if (variant == model_variant::gemma4_multiwidth) {
      build_gemma4_multiwidth_prepared_model(prepared);
    } else {
      build_prepared_model(prepared);
    }
    if (variant == model_variant::flash_kv_width_mismatch) {
      apply_flash_kv_width_mismatch(prepared);
    }
    const emel::model::data & model = stabilize_model(prepared);
    const auto matmul_policy =
        emel::text::generator::matmul::make_auto_execution_policy(
            parallel_matmul_lanes);
    generator = std::make_unique<emel::text::generator::sm>(
        emel::text::generator::dependencies{
            .model = model,
            .conditioner = conditioner,
            .matmul_policy = matmul_policy,
            .runtime_policy =
                emel::text::generator::test::make_auto_runtime_policy(model),
            .formatter_ctx = formatter_ctx,
            .format_prompt = format_prompt,
            .kv_cache = kv_cache,
        });
    hello_id = prepared.hello_id;
    world_id = prepared.world_id;
  }

  emel::text::generator::event::initialize make_initialize(
      callback_tracker & tracker,
      emel::error::type * error_out = nullptr,
      const emel::text::generator::selection_mode selection_mode =
          emel::text::generator::selection_mode::sample_logits) {
    const std::span<emel::logits::sampler::fn> sampler_span =
        selection_mode == emel::text::generator::selection_mode::sample_logits
            ? std::span<emel::logits::sampler::fn>{samplers}
            : std::span<emel::logits::sampler::fn>{};
    emel::text::generator::event::initialize request{
      &tokenizer,
      tokenizer_bind_dispatch,
      tokenizer_tokenize_dispatch,
      sampler_span,
    };
    request.preprocessor_variant = emel::text::tokenizer::preprocessor::preprocessor_kind::bpe;
    request.encoder_variant = emel::text::encoders::encoder_kind::bpe;
    request.add_special = false;
    request.parse_special = false;
    request.selection_mode = selection_mode;
    request.max_prompt_tokens = 8;
    request.max_generated_tokens = 4;
    request.max_blocks = 2;
    request.block_tokens = 4;
    request.strip_leading_space = false;
    request.error_out = error_out;
    request.on_done = emel::callback<void(const emel::text::generator::events::initialize_done &)>(
        &tracker, on_initialize_done);
    request.on_error = emel::callback<void(const emel::text::generator::events::initialize_error &)>(
        &tracker, on_initialize_error);
    return request;
  }

  emel::text::generator::event::generate make_generate(callback_tracker & tracker,
                                                 char * output,
                                                 size_t output_capacity,
                                                 size_t & output_length_out,
                                                 emel::error::type * error_out = nullptr) {
    emel::text::generator::event::generate request{
      std::span<const emel::text::formatter::chat_message>{k_phase_4_messages},
      k_phase_4_max_tokens,
      std::span<char>{output, output_capacity},
      output_length_out,
    };
    request.add_generation_prompt = k_phase_4_add_generation_prompt;
    request.enable_thinking = k_phase_4_enable_thinking;
    request.error_out = error_out;
    request.on_done = emel::callback<void(const emel::text::generator::events::generation_done &)>(
        &tracker, on_generate_done);
    request.on_error = emel::callback<void(const emel::text::generator::events::generation_error &)>(
        &tracker, on_generate_error);
    return request;
  }

  emel::model::data & model() noexcept { return prepared.data; }
};

}  // namespace

TEST_CASE("generator_starts_uninitialized") {
  auto fixture = std::make_unique<generator_fixture>();
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::uninitialized>));
}

TEST_CASE("generator_initialize_reserves_lifecycle_managed_graph_tensors") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::none);
  auto initialize = fixture->make_initialize(tracker, &error);

  REQUIRE(fixture->generator->process_event(initialize));
  REQUIRE(tracker.initialize_done_called);
  REQUIRE_FALSE(tracker.initialize_error_called);

  emel::text::generator::graph_lifecycle_snapshot graph_snapshot{};
  REQUIRE(fixture->generator->process_event(
      emel::text::generator::event::capture_graph_lifecycle{graph_snapshot}));
  REQUIRE(graph_snapshot.reservation != nullptr);
  REQUIRE(graph_snapshot.reservation->lifecycle != nullptr);
  REQUIRE(graph_snapshot.reservation->lifecycle->tensor_count > 1);
  REQUIRE(graph_snapshot.first_tensor_captured);
  CHECK(graph_snapshot.first_tensor.lifecycle_state ==
        emel::graph::tensor::event::lifecycle::leaf_filled);
  REQUIRE(graph_snapshot.runtime_tensor_captured);
  CHECK(graph_snapshot.runtime_tensor.lifecycle_state ==
        emel::graph::tensor::event::lifecycle::empty);
}

TEST_CASE("generator_dependencies_accept_custom_kv_actor") {
  emel::memory::test::recording_kv_actor kv{};
  auto fixture = std::make_unique<generator_fixture>(
      generator_fixture::model_variant::canonical,
      nullptr,
      emel::text::formatter::format_raw,
      emel::memory::hybrid::bind_kv_actor(kv));
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::backend);
  const auto request = fixture->make_initialize(tracker, &error);

  REQUIRE(fixture->generator->process_event(request));
  REQUIRE(tracker.initialize_done_called);
  REQUIRE_FALSE(tracker.initialize_error_called);
  CHECK(error == emel::error::cast(emel::text::generator::error::none));
  CHECK(kv.reserve_count == 1);

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 99;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::backend);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  REQUIRE(fixture->generator->process_event(generate_request));
  REQUIRE(generate_tracker.generate_done_called);
  REQUIRE_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(kv.allocate_sequence_count == 1);
  CHECK(kv.allocate_slots_count == 1);
  CHECK(kv.capture_view_count >= 1);
}

TEST_CASE("generator_rejects_generate_before_initialize") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  std::array<char, 16> output = {};
  size_t output_length = 7;
  emel::error::type error = emel::error::cast(emel::text::generator::error::none);
  const auto request =
      fixture->make_generate(tracker, output.data(), output.size(), output_length, &error);

  CHECK_FALSE(fixture->generator->process_event(request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::uninitialized>));
  CHECK_FALSE(tracker.generate_done_called);
  CHECK(tracker.generate_error_called);
  CHECK(error == emel::error::cast(emel::text::generator::error::invalid_request));
  CHECK(tracker.err == emel::error::cast(emel::text::generator::error::invalid_request));
}

TEST_CASE("generator_initialize_succeeds_and_enters_ready") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::backend);
  const auto request = fixture->make_initialize(tracker, &error);

  CHECK(fixture->generator->process_event(request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK(tracker.initialize_done_called);
  CHECK_FALSE(tracker.initialize_error_called);
  CHECK(error == emel::error::cast(emel::text::generator::error::none));
}

TEST_CASE("generator_reinitialize_reprepares_backend_when_block_geometry_changes") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker first_tracker{};
  emel::error::type first_error =
      emel::error::cast(emel::text::generator::error::backend);
  const auto first_request = fixture->make_initialize(first_tracker, &first_error);
  REQUIRE(fixture->generator->process_event(first_request));
  REQUIRE(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  REQUIRE(first_error == emel::error::cast(emel::text::generator::error::none));

  callback_tracker second_tracker{};
  emel::error::type second_error =
      emel::error::cast(emel::text::generator::error::backend);
  auto second_request = fixture->make_initialize(second_tracker, &second_error);
  second_request.max_blocks = 4;
  second_request.block_tokens = 2;

  CHECK(fixture->generator->process_event(second_request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK(second_tracker.initialize_done_called);
  CHECK_FALSE(second_tracker.initialize_error_called);
  CHECK(second_error == emel::error::cast(emel::text::generator::error::none));
}

TEST_CASE("generator_initialize_accepts_explicit_preselected_argmax_mode_without_sampler_chain") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::backend);
  const auto request = fixture->make_initialize(
      tracker, &error, emel::text::generator::selection_mode::preselected_argmax);

  CHECK(fixture->generator->process_event(request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK(tracker.initialize_done_called);
  CHECK_FALSE(tracker.initialize_error_called);
  CHECK(error == emel::error::cast(emel::text::generator::error::none));
}

TEST_CASE("generator_rejects_invalid_initialize_request") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::none);
  auto request = fixture->make_initialize(tracker, &error);
  request.tokenizer_sm = nullptr;

  CHECK_FALSE(fixture->generator->process_event(request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::uninitialized>));
  CHECK_FALSE(tracker.initialize_done_called);
  CHECK(tracker.initialize_error_called);
  CHECK(error == emel::error::cast(emel::text::generator::error::invalid_request));
  CHECK(tracker.err == emel::error::cast(emel::text::generator::error::invalid_request));
}

TEST_CASE("generator_initialize_rejects_block_pool_exceeding_physical_capacity") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::none);
  auto request = fixture->make_initialize(tracker, &error);
  // Fixture model n_ctx is 8 (physical capacity 8 with 4-token blocks); a
  // 3-block pool could hand out ids beyond physical storage, so the geometry
  // guard must route to invalid_request before reserve.
  request.max_blocks = 3;
  request.block_tokens = 4;

  CHECK_FALSE(fixture->generator->process_event(request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::uninitialized>));
  CHECK_FALSE(tracker.initialize_done_called);
  CHECK(tracker.initialize_error_called);
  CHECK(error == emel::error::cast(emel::text::generator::error::invalid_request));
  CHECK(tracker.err == emel::error::cast(emel::text::generator::error::invalid_request));
}

TEST_CASE("generator_initialize_requires_explicit_block_geometry") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::none);
  auto request = fixture->make_initialize(tracker, &error);
  // The public contract requires explicit positive geometry; zero fields are
  // rejected by valid_initialize before any backend or memory dispatch.
  request.max_blocks = 0;
  request.block_tokens = 0;

  CHECK_FALSE(fixture->generator->process_event(request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::uninitialized>));
  CHECK_FALSE(tracker.initialize_done_called);
  CHECK(tracker.initialize_error_called);
  CHECK(error == emel::error::cast(emel::text::generator::error::invalid_request));
}

TEST_CASE("generator_initialize_rejects_block_tokens_larger_than_context") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::none);
  auto request = fixture->make_initialize(tracker, &error);
  request.max_blocks = 1;
  request.block_tokens = INT32_MAX;

  CHECK_FALSE(fixture->generator->process_event(request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::uninitialized>));
  CHECK_FALSE(tracker.initialize_done_called);
  CHECK(tracker.initialize_error_called);
  CHECK(error == emel::error::cast(emel::text::generator::error::invalid_request));
  CHECK(tracker.err == emel::error::cast(emel::text::generator::error::invalid_request));
}

TEST_CASE("generator_requires_construction_time_dependencies") {
  CHECK_FALSE(std::is_default_constructible_v<emel::text::generator::sm>);
}

TEST_CASE("generator_initialize_reports_original_request_without_generation_callbacks") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::backend);
  const auto request = fixture->make_initialize(tracker, &error);

  REQUIRE(fixture->generator->process_event(request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK(tracker.initialize_done_called);
  CHECK_FALSE(tracker.initialize_error_called);
  CHECK(tracker.initialize_request == &request);
  CHECK_FALSE(tracker.generate_done_called);
  CHECK_FALSE(tracker.generate_error_called);
  CHECK(tracker.generate_request == nullptr);
  CHECK(error == emel::error::cast(emel::text::generator::error::none));
}

TEST_CASE("generator_initialize_can_rebind_ready_session_without_re_reserving_graph") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker first_tracker{};
  emel::error::type first_error = emel::error::cast(emel::text::generator::error::backend);
  const auto first_request = fixture->make_initialize(first_tracker, &first_error);

  REQUIRE(fixture->generator->process_event(first_request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK(first_tracker.initialize_done_called);
  CHECK(first_error == emel::error::cast(emel::text::generator::error::none));

  callback_tracker second_tracker{};
  emel::error::type second_error = emel::error::cast(emel::text::generator::error::backend);
  const auto second_request = fixture->make_initialize(second_tracker, &second_error);

  CHECK(fixture->generator->process_event(second_request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK(second_tracker.initialize_done_called);
  CHECK_FALSE(second_tracker.initialize_error_called);
  CHECK(second_error == emel::error::cast(emel::text::generator::error::none));
}

TEST_CASE("generator_generate_runs_native_generator_contract") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator->process_event(initialize_request));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 99;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::backend);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  CHECK(fixture->generator->process_event(generate_request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_done_called);
  CHECK(generate_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(generate_tracker.tokens_generated == 1);
  CHECK(output_length == 5);
  CHECK(generate_tracker.output_length == 5);
  CHECK(std::string_view(output.data(), output_length) == "world");
  const auto diagnostics = capture_generator_diagnostics(*fixture->generator);
  CHECK(diagnostics.kernel_dispatch_calls > 0u);
  CHECK(diagnostics.flash_attention_dispatch_calls > 0u);
  if (host_is_aarch64() || host_is_x86_64()) {
    CHECK(diagnostics.optimized_flash_dispatch_calls > 0u);
    CHECK(diagnostics.shared_flash_dispatch_calls == 0u);
  } else {
    CHECK(diagnostics.optimized_flash_dispatch_calls == 0u);
    CHECK(diagnostics.shared_flash_dispatch_calls == 0u);
  }
}

TEST_CASE("generator_qwen3_generator_initializes_and_generates_one_token") {
  auto fixture = std::make_unique<generator_fixture>(generator_fixture::model_variant::qwen3_canonical);
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);

  REQUIRE(fixture->generator->process_event(initialize_request));
  REQUIRE(initialize_tracker.initialize_done_called);
  REQUIRE_FALSE(initialize_tracker.initialize_error_called);
  REQUIRE(initialize_error == emel::error::cast(emel::text::generator::error::none));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 0u;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::backend);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  CHECK(fixture->generator->process_event(generate_request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_done_called);
  CHECK(generate_tracker.tokens_generated == 1);
  CHECK(generate_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(output_length > 0u);
}

TEST_CASE("generator_gemma4_generator_initializes_and_generates_one_token") {
  auto fixture = std::make_unique<generator_fixture>(generator_fixture::model_variant::gemma4_canonical);
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);

  REQUIRE(fixture->generator->process_event(initialize_request));
  REQUIRE(initialize_tracker.initialize_done_called);
  REQUIRE_FALSE(initialize_tracker.initialize_error_called);
  REQUIRE(initialize_error == emel::error::cast(emel::text::generator::error::none));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 0u;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::backend);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  CHECK(fixture->generator->process_event(generate_request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_done_called);
  CHECK(generate_tracker.tokens_generated == 1);
  CHECK(generate_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(output_length > 0u);
}

TEST_CASE("generator_gemma4_generator_initializes_with_mixed_sliding_and_full_attention_widths") {
  auto fixture = std::make_unique<generator_fixture>(
      generator_fixture::model_variant::gemma4_multiwidth);

  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);

  REQUIRE(fixture->generator->process_event(initialize_request));
  CHECK(initialize_error == emel::error::cast(emel::text::generator::error::none));
}

TEST_CASE("generator_generate_f32_fixture_does_not_claim_quantized_optimized_dispatch") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator->process_event(initialize_request));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 0;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::backend);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  REQUIRE(fixture->generator->process_event(generate_request));
  CHECK(generate_error == emel::error::cast(emel::text::generator::error::none));
  const auto diagnostics = capture_generator_diagnostics(*fixture->generator);
  CHECK(diagnostics.optimized_q2_dispatch_calls == 0u);
  CHECK(diagnostics.shared_q2_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_q3_dispatch_calls == 0u);
  CHECK(diagnostics.shared_q3_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_q4_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_q4_vector_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_q4_vector_packed_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_q4_vector_packed_q8_rhs_dispatch_calls == 0u);
  CHECK(diagnostics.shared_q4_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_q6_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_q6_vector_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_q6_vector_packed_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_q6_vector_packed_q8_rhs_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_dispatch_calls == 0u);
  CHECK(diagnostics.shared_q6_dispatch_calls == 0u);
}

TEST_CASE("generator_quantized_path_audit_classifies_canonical_stage_families") {
  auto prepared = std::make_unique<prepared_model>();
  build_prepared_model(*prepared);

  REQUIRE(find_tensor(*prepared, "token_embd.weight") != nullptr);
  REQUIRE(find_tensor(*prepared, "output_norm.weight") != nullptr);
  REQUIRE(find_tensor(*prepared, "output.weight") != nullptr);
  REQUIRE(find_tensor(*prepared, "blk.0.attn_q.weight") != nullptr);
  REQUIRE(find_tensor(*prepared, "blk.0.ffn_norm.weight") != nullptr);

  find_tensor(*prepared, "token_embd.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q2_k);
  find_tensor(*prepared, "output.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q3_k);
  find_tensor(*prepared, "blk.0.attn_q.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q6_k);

  emel::model::llama::detail::execution_view execution{};
  REQUIRE(emel::model::llama::detail::build_execution_view(stabilize_model(*prepared), execution) ==
          emel::error::cast(emel::model::loader::error::none));

  const auto audit = emel::model::llama::detail::build_quantized_path_audit(execution);
  const auto & token_embedding = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::token_embedding);
  const auto & output_norm = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::output_norm);
  const auto & output = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::output);
  const auto & attention_q = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::attention_q);
  const auto & feed_forward_norm = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::feed_forward_norm);

  CHECK(token_embedding.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
  CHECK(output_norm.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
  CHECK(output.contract ==
        emel::model::llama::detail::quantized_contract_kind::native_quantized);
  CHECK(attention_q.contract ==
        emel::model::llama::detail::quantized_contract_kind::native_quantized);
  CHECK(feed_forward_norm.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
}

TEST_CASE("generator_qwen3_quantized_path_audit_publishes_explicit_norm_stage_families") {
  auto prepared = std::make_unique<prepared_model>();
  build_qwen3_quantized_contract_prepared_model(*prepared, true, true);
  apply_qwen3_quantized_contract_tensor_types(*prepared);

  emel::model::llama::detail::execution_view execution{};
  REQUIRE(emel::model::llama::detail::build_execution_view(stabilize_model(*prepared), execution) ==
          emel::error::cast(emel::model::loader::error::none));

  const auto audit = emel::model::llama::detail::build_quantized_path_audit(execution);
  const auto & attention_q_norm = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::attention_q_norm);
  const auto & attention_k_norm = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::attention_k_norm);
  const auto & attention_q = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::attention_q);

  CHECK(attention_q_norm.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
  CHECK(attention_k_norm.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
  CHECK(attention_q.contract ==
        emel::model::llama::detail::quantized_contract_kind::native_quantized);
}

TEST_CASE("generator_qwen3_quantized_path_audit_requires_attention_q_norm_tensor") {
  auto prepared = std::make_unique<prepared_model>();
  build_qwen3_quantized_contract_prepared_model(*prepared, false, true);

  emel::model::llama::detail::execution_view execution{};
  CHECK(emel::model::llama::detail::build_execution_view(stabilize_model(*prepared), execution) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("generator_qwen3_quantized_path_audit_requires_attention_k_norm_tensor") {
  auto prepared = std::make_unique<prepared_model>();
  build_qwen3_quantized_contract_prepared_model(*prepared, true, false);

  emel::model::llama::detail::execution_view execution{};
  CHECK(emel::model::llama::detail::build_execution_view(stabilize_model(*prepared), execution) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("generator_quantized_path_audit_marks_unsupported_quantized_stage_no_claim") {
  auto prepared = std::make_unique<prepared_model>();
  build_prepared_model(*prepared);

  REQUIRE(find_tensor(*prepared, "blk.0.attn_q.weight") != nullptr);
  find_tensor(*prepared, "blk.0.attn_q.weight")->type = emel::kernel::detail::dtype_q5_1;

  emel::model::llama::detail::execution_view execution{};
  REQUIRE(emel::model::llama::detail::build_execution_view(stabilize_model(*prepared), execution) ==
          emel::error::cast(emel::model::loader::error::none));

  const auto audit = emel::model::llama::detail::build_quantized_path_audit(execution);
  const auto & attention_q = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::attention_q);

  CHECK(attention_q.tensor_type == emel::kernel::detail::dtype_q5_1);
  CHECK(attention_q.contract ==
        emel::model::llama::detail::quantized_contract_kind::explicit_no_claim);
  CHECK(attention_q.consistent_across_layers);
}

TEST_CASE("generator_quantized_path_audit_marks_nonvector_f32_stage_disallowed_fallback") {
  auto prepared = std::make_unique<prepared_model>();
  build_prepared_model(*prepared);

  REQUIRE(find_tensor(*prepared, "token_embd.weight") != nullptr);
  REQUIRE(find_tensor(*prepared, "output.weight") != nullptr);
  REQUIRE(find_tensor(*prepared, "blk.0.attn_q.weight") != nullptr);

  find_tensor(*prepared, "token_embd.weight")->type = emel::kernel::detail::dtype_f32;
  find_tensor(*prepared, "output.weight")->type = emel::kernel::detail::dtype_f32;
  find_tensor(*prepared, "blk.0.attn_q.weight")->type = emel::kernel::detail::dtype_f32;

  emel::model::llama::detail::execution_view execution{};
  REQUIRE(emel::model::llama::detail::build_execution_view(stabilize_model(*prepared), execution) ==
          emel::error::cast(emel::model::loader::error::none));

  const auto audit = emel::model::llama::detail::build_quantized_path_audit(execution);
  const auto & token_embedding = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::token_embedding);
  const auto & output = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::output);
  const auto & attention_q = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::attention_q);

  CHECK(token_embedding.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
  CHECK(output.contract ==
        emel::model::llama::detail::quantized_contract_kind::disallowed_fallback);
  CHECK(attention_q.contract ==
        emel::model::llama::detail::quantized_contract_kind::disallowed_fallback);
}

TEST_CASE("generator_lfm2_quantized_path_audit_accepts_hybrid_native_quantized_mix") {
  auto prepared = std::make_unique<prepared_model>();
  build_lfm2_quantized_contract_prepared_model(*prepared);

  find_tensor(*prepared, "token_embd.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q6_k);
  find_tensor(*prepared, "blk.0.ffn_gate.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q4_k);
  find_tensor(*prepared, "blk.1.ffn_gate.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q4_k);
  find_tensor(*prepared, "blk.2.ffn_gate.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q4_k);
  find_tensor(*prepared, "blk.0.ffn_down.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q6_k);
  find_tensor(*prepared, "blk.1.ffn_down.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q4_k);
  find_tensor(*prepared, "blk.2.ffn_down.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q6_k);
  find_tensor(*prepared, "blk.0.ffn_up.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q4_k);
  find_tensor(*prepared, "blk.1.ffn_up.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q4_k);
  find_tensor(*prepared, "blk.2.ffn_up.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q4_k);
  find_tensor(*prepared, "blk.2.attn_q.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q4_k);
  find_tensor(*prepared, "blk.2.attn_k.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q4_k);
  find_tensor(*prepared, "blk.2.attn_v.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q6_k);
  find_tensor(*prepared, "blk.2.attn_output.weight")->type =
      static_cast<int32_t>(emel::kernel::event::dtype::q4_k);

  emel::model::llama::detail::execution_view execution{};
  REQUIRE(emel::model::llama::detail::build_execution_view(stabilize_model(*prepared), execution) ==
          emel::error::cast(emel::model::loader::error::none));

  const auto audit = emel::model::llama::detail::build_quantized_path_audit(execution);
  const auto & token_embedding = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::token_embedding);
  const auto & attention_q = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::attention_q);
  const auto & attention_q_norm = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::attention_q_norm);
  const auto & feed_forward_down = find_stage_audit(
      audit, emel::model::llama::detail::quantized_stage_family::feed_forward_down);

  CHECK(token_embedding.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
  CHECK(attention_q.contract ==
        emel::model::llama::detail::quantized_contract_kind::native_quantized);
  CHECK(attention_q.consistent_across_layers);
  CHECK(attention_q_norm.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
  CHECK(feed_forward_down.contract ==
        emel::model::llama::detail::quantized_contract_kind::native_quantized);
  CHECK_FALSE(feed_forward_down.consistent_across_layers);
}

TEST_CASE("generator_initialize_quantized_contract_fixture_reports_zero_disallowed_fallback_stages") {
  auto fixture = std::make_unique<generator_fixture>(
      generator_fixture::model_variant::quantized_contract);
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize = fixture->make_initialize(tracker, &error);

  REQUIRE(fixture->generator->process_event(initialize));
  CHECK(error == emel::error::cast(emel::text::generator::error::none));
  const auto diagnostics = capture_generator_diagnostics(*fixture->generator);
  CHECK(diagnostics.native_quantized_stage_count == 8u);
  CHECK(diagnostics.approved_dense_f32_stage_count == 4u);
  CHECK(diagnostics.disallowed_fallback_stage_count == 0u);
  CHECK(diagnostics.explicit_no_claim_stage_count == 0u);
}

TEST_CASE("generator_generate_quantized_contract_fixture_preserves_zero_disallowed_fallback") {
  auto fixture = std::make_unique<generator_fixture>(
      generator_fixture::model_variant::quantized_contract);
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);

  REQUIRE(fixture->generator->process_event(initialize_request));
  REQUIRE(initialize_error == emel::error::cast(emel::text::generator::error::none));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 0;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::backend);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  REQUIRE(fixture->generator->process_event(generate_request));
  CHECK_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_done_called);
  CHECK(generate_error == emel::error::cast(emel::text::generator::error::none));
  const auto diagnostics = capture_generator_diagnostics(*fixture->generator);
  CHECK(diagnostics.native_quantized_stage_count == 8u);
  CHECK(diagnostics.approved_dense_f32_stage_count == 4u);
  CHECK(diagnostics.disallowed_fallback_stage_count == 0u);
  CHECK(diagnostics.explicit_no_claim_stage_count == 0u);
  if (host_is_aarch64()) {
    CHECK(diagnostics.optimized_q6_vector_dispatch_calls > 0u);
    CHECK(diagnostics.optimized_q6_vector_packed_dispatch_calls > 0u);
    CHECK(diagnostics.optimized_q6_vector_packed_q8_rhs_dispatch_calls > 0u);
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_dispatch_calls > 0u);
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_calls > 0u);
#else
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_dispatch_calls == 0u);
#endif
  } else if (host_is_x86_64()) {
    CHECK(diagnostics.optimized_q2_dispatch_calls > 0u);
    CHECK(diagnostics.shared_q2_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q3_dispatch_calls > 0u);
    CHECK(diagnostics.shared_q3_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_dispatch_calls > 0u);
    CHECK(diagnostics.shared_q6_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_packed_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_packed_q8_rhs_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_calls == 0u);
  } else {
    CHECK(diagnostics.optimized_q6_vector_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_packed_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_packed_q8_rhs_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_calls == 0u);
  }
}

TEST_CASE("generator_generate_quantized_contract_fixture_supports_explicit_preselected_argmax_mode") {
  auto fixture = std::make_unique<generator_fixture>(
      generator_fixture::model_variant::quantized_contract);
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(
      initialize_tracker,
      &initialize_error,
      emel::text::generator::selection_mode::preselected_argmax);
  auto long_initialize_request = initialize_request;
  long_initialize_request.max_generated_tokens = 16;
  long_initialize_request.max_blocks = 2;

  REQUIRE(fixture->generator->process_event(long_initialize_request));
  REQUIRE(initialize_error == emel::error::cast(emel::text::generator::error::none));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 0;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::backend);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  REQUIRE(fixture->generator->process_event(generate_request));
  CHECK_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_done_called);
  CHECK(generate_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(output_length > 0u);
  const auto diagnostics = capture_generator_diagnostics(*fixture->generator);
  if (host_is_aarch64()) {
    CHECK(diagnostics.optimized_q6_vector_argmax_dispatch_calls > 0u);
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    CHECK(diagnostics.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_calls > 0u);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    CHECK(diagnostics.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_calls > 0u);
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_calls == 0u);
#else
    CHECK(diagnostics.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_calls == 0u);
#endif
  } else {
    CHECK(diagnostics.optimized_q6_vector_argmax_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_calls == 0u);
    CHECK(diagnostics.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_calls == 0u);
  }
}

TEST_CASE(
    "generator_generate_quantized_contract_fixture_allows_repeated_explicit_preselected_argmax_generates") {
  auto fixture = std::make_unique<generator_fixture>(
      generator_fixture::model_variant::quantized_contract);
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(
      initialize_tracker,
      &initialize_error,
      emel::text::generator::selection_mode::preselected_argmax);

  REQUIRE(fixture->generator->process_event(initialize_request));
  REQUIRE(initialize_error == emel::error::cast(emel::text::generator::error::none));

  callback_tracker first_tracker{};
  std::array<char, 64> first_output = {};
  size_t first_output_length = 0;
  emel::error::type first_error = emel::error::cast(emel::text::generator::error::backend);
  const auto first_request = fixture->make_generate(
      first_tracker, first_output.data(), first_output.size(), first_output_length, &first_error);

  REQUIRE(fixture->generator->process_event(first_request));
  REQUIRE_FALSE(first_tracker.generate_error_called);
  REQUIRE(first_tracker.generate_done_called);
  REQUIRE(first_error == emel::error::cast(emel::text::generator::error::none));
  REQUIRE(first_output_length > 0u);

  callback_tracker second_tracker{};
  std::array<char, 64> second_output = {};
  size_t second_output_length = 0;
  emel::error::type second_error = emel::error::cast(emel::text::generator::error::backend);
  const auto second_request = fixture->make_generate(
      second_tracker,
      second_output.data(),
      second_output.size(),
      second_output_length,
      &second_error);

  CHECK(fixture->generator->process_event(second_request));
  CHECK_FALSE(second_tracker.generate_error_called);
  CHECK(second_tracker.generate_done_called);
  CHECK(second_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(second_output_length > 0u);
}

TEST_CASE(
    "generator_generate_quantized_contract_fixture_allows_repeated_multi_token_explicit_preselected_argmax_generates") {
  auto fixture = std::make_unique<generator_fixture>(
      generator_fixture::model_variant::quantized_contract);
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(
      initialize_tracker,
      &initialize_error,
      emel::text::generator::selection_mode::preselected_argmax);

  REQUIRE(fixture->generator->process_event(initialize_request));
  REQUIRE(initialize_error == emel::error::cast(emel::text::generator::error::none));

  const auto make_long_generate =
      [](callback_tracker & tracker,
         std::array<char, 256> & output,
         size_t & output_length,
         emel::error::type * error_out) {
        emel::text::generator::event::generate request{
          std::span<const emel::text::formatter::chat_message>{
              generator_fixture::k_phase_4_messages},
          2,
          std::span<char>{output.data(), output.size()},
          output_length,
        };
        request.add_generation_prompt = generator_fixture::k_phase_4_add_generation_prompt;
        request.enable_thinking = generator_fixture::k_phase_4_enable_thinking;
        request.error_out = error_out;
        request.on_done = emel::callback<void(const emel::text::generator::events::generation_done &)>(
            &tracker, on_generate_done);
        request.on_error = emel::callback<void(const emel::text::generator::events::generation_error &)>(
            &tracker, on_generate_error);
        return request;
      };

  callback_tracker first_tracker{};
  std::array<char, 256> first_output = {};
  size_t first_output_length = 0;
  emel::error::type first_error = emel::error::cast(emel::text::generator::error::backend);
  const auto first_request =
      make_long_generate(first_tracker, first_output, first_output_length, &first_error);

  REQUIRE(fixture->generator->process_event(first_request));
  REQUIRE_FALSE(first_tracker.generate_error_called);
  REQUIRE(first_tracker.generate_done_called);
  REQUIRE(first_error == emel::error::cast(emel::text::generator::error::none));
  REQUIRE(first_output_length > 0u);

  callback_tracker second_tracker{};
  std::array<char, 256> second_output = {};
  size_t second_output_length = 0;
  emel::error::type second_error = emel::error::cast(emel::text::generator::error::backend);
  const auto second_request =
      make_long_generate(second_tracker, second_output, second_output_length, &second_error);

  CHECK(fixture->generator->process_event(second_request));
  CHECK_FALSE(second_tracker.generate_error_called);
  CHECK(second_tracker.generate_done_called);
  CHECK(second_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(second_output_length > 0u);
}

TEST_CASE("generator structured message request pins the phase 4 contract") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator->process_event(initialize_request));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 0;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::backend);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  CHECK(generate_request.messages.size() == generator_fixture::k_phase_4_messages.size());
  CHECK(generate_request.messages[0].role == generator_fixture::k_phase_4_messages[0].role);
  CHECK(generate_request.messages[0].content ==
        generator_fixture::k_phase_4_messages[0].content);
  CHECK(generate_request.add_generation_prompt ==
        generator_fixture::k_phase_4_add_generation_prompt);
  CHECK(generate_request.enable_thinking ==
        generator_fixture::k_phase_4_enable_thinking);
  CHECK(generate_request.max_tokens == generator_fixture::k_phase_4_max_tokens);
  CHECK(generate_request.output.data() == output.data());
  CHECK(generate_request.output.size() == output.size());
  CHECK(fixture->generator->process_event(generate_request));
  CHECK_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_done_called);
  CHECK(generate_tracker.generate_request == &generate_request);
  CHECK(generate_tracker.tokens_generated == generator_fixture::k_phase_4_max_tokens);
  CHECK(generate_tracker.output_length == 5);
  CHECK(output_length == generate_tracker.output_length);
  CHECK(std::string_view(output.data(), output_length) == "world");
}

TEST_CASE("generator structured message request reaches the conditioner formatter") {
  static constexpr std::array<emel::text::formatter::chat_message, 2> k_messages = {
      emel::text::formatter::chat_message{
          .role = "system",
          .content = "policy",
      },
      emel::text::formatter::chat_message{
          .role = "user",
          .content = "hello",
      },
  };
  checked_formatter_ctx formatter_ctx{
      .expected_messages = std::span<const emel::text::formatter::chat_message>{k_messages},
      .expected_add_generation_prompt = true,
      .expected_enable_thinking = false,
      .formatted = "hello",
      .seen = false,
  };
  auto fixture = std::make_unique<generator_fixture>(generator_fixture::model_variant::canonical,
                                                     &formatter_ctx,
                                                     format_checked_messages);
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator->process_event(initialize_request));
  REQUIRE(initialize_error == emel::error::cast(emel::text::generator::error::none));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 0;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::backend);
  emel::text::generator::event::generate generate_request{
      std::span<const emel::text::formatter::chat_message>{k_messages},
      1,
      std::span<char>{output.data(), output.size()},
      output_length,
  };
  generate_request.add_generation_prompt = true;
  generate_request.enable_thinking = false;
  generate_request.error_out = &generate_error;
  generate_request.on_done = emel::callback<void(const emel::text::generator::events::generation_done &)>(
      &generate_tracker, on_generate_done);
  generate_request.on_error =
      emel::callback<void(const emel::text::generator::events::generation_error &)>(
          &generate_tracker, on_generate_error);

  REQUIRE(fixture->generator->process_event(generate_request));
  CHECK(formatter_ctx.seen);
  CHECK_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_done_called);
  CHECK(generate_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(generate_tracker.tokens_generated == 1);
  CHECK(std::string_view(output.data(), output_length) == "world");
}

TEST_CASE("generator_generate_reports_bounded_output_buffer_errors") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator->process_event(initialize_request));

  callback_tracker generate_tracker{};
  std::array<char, 4> output = {};
  size_t output_length = 17;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::none);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  CHECK_FALSE(fixture->generator->process_event(generate_request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK_FALSE(generate_tracker.generate_done_called);
  CHECK(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_request == &generate_request);
  CHECK(generate_tracker.tokens_generated == 0);
  CHECK(generate_tracker.output_length == 0);
  CHECK(output_length == 0);
  CHECK(generate_error == emel::error::cast(emel::text::generator::error::invalid_request));
  CHECK(generate_tracker.err == emel::error::cast(emel::text::generator::error::invalid_request));
}

TEST_CASE("generator_generate_uses_nonflash_runtime_without_claiming_flash") {
  auto fixture = std::make_unique<generator_fixture>(
      generator_fixture::model_variant::flash_kv_width_mismatch);
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator->process_event(initialize_request));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 0;
  emel::error::type generate_error = emel::error::cast(emel::text::generator::error::none);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  CHECK(fixture->generator->process_event(generate_request));
  CHECK(fixture->generator->is(stateforward::sml::state<emel::text::generator::ready>));
  CHECK(generate_tracker.generate_done_called);
  CHECK_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(generate_tracker.tokens_generated == 1);
  CHECK(output_length == 5);
  CHECK(generate_tracker.output_length == 5);
  CHECK(std::string_view(output.data(), output_length) == "world");
  const auto diagnostics = capture_generator_diagnostics(*fixture->generator);
  CHECK(diagnostics.flash_attention_dispatch_calls == 0u);
  CHECK(diagnostics.optimized_flash_dispatch_calls == 0u);
  CHECK(diagnostics.shared_flash_dispatch_calls == 0u);
}

TEST_CASE("generator_generate_multiple_tokens_and_resets_sequence_on_reuse") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator->process_event(initialize_request));

  callback_tracker first_tracker{};
  std::array<char, 32> first_output = {};
  size_t first_output_length = 0;
  emel::error::type first_error = emel::error::cast(emel::text::generator::error::backend);
  auto first_request =
      fixture->make_generate(first_tracker, first_output.data(), first_output.size(),
                             first_output_length, &first_error);
  first_request.max_tokens = 2;

  CHECK(fixture->generator->process_event(first_request));
  CHECK(first_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(first_tracker.tokens_generated == 2);
  CHECK(std::string_view(first_output.data(), first_output_length) == "worldworld");

  callback_tracker second_tracker{};
  std::array<char, 16> second_output = {};
  size_t second_output_length = 0;
  emel::error::type second_error = emel::error::cast(emel::text::generator::error::backend);
  const auto second_request =
      fixture->make_generate(second_tracker, second_output.data(), second_output.size(),
                             second_output_length, &second_error);

  CHECK(fixture->generator->process_event(second_request));
  CHECK(second_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(second_tracker.tokens_generated == 1);
  CHECK(std::string_view(second_output.data(), second_output_length) == "world");
}

TEST_CASE("generator_resets_renderer_stop_state_when_reusing_a_sequence") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::text::generator::error::backend);
  auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  const std::array<std::string_view, 1> stops = {"worldworld"};
  initialize_request.stop_sequences = stops;
  REQUIRE(fixture->generator->process_event(initialize_request));

  callback_tracker first_tracker{};
  std::array<char, 32> first_output = {};
  size_t first_output_length = 0;
  emel::error::type first_error = emel::error::cast(emel::text::generator::error::backend);
  auto first_request =
      fixture->make_generate(first_tracker, first_output.data(), first_output.size(),
                             first_output_length, &first_error);
  first_request.max_tokens = 2;

  REQUIRE(fixture->generator->process_event(first_request));
  CHECK(first_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(first_tracker.tokens_generated == 2);
  CHECK(first_output_length == 0);

  callback_tracker second_tracker{};
  std::array<char, 32> second_output = {};
  size_t second_output_length = 0;
  emel::error::type second_error = emel::error::cast(emel::text::generator::error::backend);
  auto second_request =
      fixture->make_generate(second_tracker, second_output.data(), second_output.size(),
                             second_output_length, &second_error);
  second_request.max_tokens = 2;

  CHECK(fixture->generator->process_event(second_request));
  CHECK(second_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(second_tracker.tokens_generated == 2);
  CHECK(second_output_length == 0);
}

TEST_CASE("generator_reinitialize_clears_lifecycle_publish_state_before_next_generate") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker first_initialize_tracker{};
  emel::error::type first_initialize_error =
      emel::error::cast(emel::text::generator::error::backend);
  const auto first_initialize =
      fixture->make_initialize(first_initialize_tracker, &first_initialize_error);
  REQUIRE(fixture->generator->process_event(first_initialize));

  callback_tracker first_generate_tracker{};
  std::array<char, 32> first_output = {};
  size_t first_output_length = 0;
  emel::error::type first_generate_error = emel::error::cast(emel::text::generator::error::backend);
  const auto first_generate =
      fixture->make_generate(first_generate_tracker, first_output.data(), first_output.size(),
                             first_output_length, &first_generate_error);
  REQUIRE(fixture->generator->process_event(first_generate));
  REQUIRE(first_generate_error == emel::error::cast(emel::text::generator::error::none));

  callback_tracker second_initialize_tracker{};
  emel::error::type second_initialize_error =
      emel::error::cast(emel::text::generator::error::backend);
  const auto second_initialize =
      fixture->make_initialize(second_initialize_tracker, &second_initialize_error);
  REQUIRE(fixture->generator->process_event(second_initialize));
  REQUIRE(second_initialize_error == emel::error::cast(emel::text::generator::error::none));

  callback_tracker second_generate_tracker{};
  std::array<char, 32> second_output = {};
  size_t second_output_length = 0;
  emel::error::type second_generate_error = emel::error::cast(emel::text::generator::error::backend);
  const auto second_generate =
      fixture->make_generate(second_generate_tracker, second_output.data(), second_output.size(),
                             second_output_length, &second_generate_error);

  CHECK(fixture->generator->process_event(second_generate));
  CHECK(second_generate_error == emel::error::cast(emel::text::generator::error::none));
  CHECK(second_generate_tracker.tokens_generated == 1);
  CHECK(std::string_view(second_output.data(), second_output_length) == "world");
}

TEST_CASE("generator_docs_table_uses_typed_completion_event_names") {
  using machine_t = stateforward::sml::sm<emel::text::generator::model>;
  using transitions = typename machine_t::transitions;

  bool has_initialize_completion = false;
  bool has_generate_completion = false;

  for_each_type(transitions{}, [&]<class transition_t>() {
    using event = typename transition_t::event;
    const std::string event_name = emel::docs::detail::table_event_name<event>();
    if (event_name == "completion<initialize_run>") {
      has_initialize_completion = true;
    }
    if (event_name == "completion<generate_run>") {
      has_generate_completion = true;
    }
  });

  CHECK(has_initialize_completion);
  CHECK(has_generate_completion);
}

TEST_CASE("generator_sm_models_explicit_prefill_boundary_and_decode_compute_states") {
  using machine_t = stateforward::sml::sm<emel::text::generator::model>;
  using states = typename machine_t::states;

  CHECK(emel::detail::type_list_contains<
        emel::text::generator::prefill_running,
        states>::value);
  CHECK(emel::detail::type_list_contains<
        emel::text::generator::prefill_result_decision,
        states>::value);
  CHECK_FALSE(emel::detail::type_list_contains<
              emel::text::generator::prefill::contract_runtime_decision,
              states>::value);
  CHECK_FALSE(emel::detail::type_list_contains<
              emel::text::generator::prefill::contract_flash_decision,
              states>::value);
  CHECK_FALSE(emel::detail::type_list_contains<
              emel::text::generator::prefill::contract_nonflash_decision,
              states>::value);
  CHECK_FALSE(emel::detail::type_list_contains<
              emel::text::generator::prefill::compute_result_decision,
              states>::value);
  CHECK(emel::detail::type_list_contains<emel::text::generator::decode_compute_flash, states>::value);
  CHECK(emel::detail::type_list_contains<
        emel::text::generator::decode_compute_nonflash,
        states>::value);
}

TEST_CASE("generator_sm_models_explicit_initializer_boundary") {
  using machine_t = stateforward::sml::sm<emel::text::generator::model>;
  using states = typename machine_t::states;

  CHECK(emel::detail::type_list_contains<emel::text::generator::initializing, states>::value);
  CHECK(emel::detail::type_list_contains<
        emel::text::generator::initializer_result_decision,
        states>::value);
  CHECK_FALSE(emel::detail::type_list_contains<
              emel::text::generator::initializer::binding_conditioner,
              states>::value);
  CHECK_FALSE(emel::detail::type_list_contains<
              emel::text::generator::initializer::configuring_sampler_decision,
              states>::value);
}

TEST_CASE("generator_initialize_public_wrapper_has_no_runtime_branch") {
  std::ifstream stream(repo_root() / "src" / "emel" / "text" / "generator" / "sm.hpp");
  REQUIRE(stream.good());

  const std::string source((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
  const std::string marker = "bool process_event(const event::initialize &";
  const auto start = source.find(marker);
  REQUIRE(start != std::string::npos);
  const auto end = source.find("bool process_event(const event::generate &", start);
  REQUIRE(end != std::string::npos);

  const auto wrapper = source.substr(start, end - start);
  CHECK(wrapper.find("if (") == std::string::npos);
  CHECK(wrapper.find(" ? ") == std::string::npos);
  CHECK(wrapper.find("switch") == std::string::npos);
}

TEST_CASE("generator_route_guards_do_not_delegate_behavior_selection_to_detail_helpers") {
  const auto read_source = [](const std::filesystem::path & path) {
    std::ifstream stream(path);
    REQUIRE(stream.good());
    return std::string(
        (std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  };

  const std::string parent_guards =
      read_source(repo_root() / "src" / "emel" / "text" / "generator" / "guards.hpp");
  const std::string prefill_guards =
      read_source(repo_root() / "src" / "emel" / "text" / "generator" / "prefill" / "guards.hpp");
  const std::array<std::string_view, 7> forbidden_parent_patterns{
    "emel::text::generator::detail::preselected_argmax_direct_supported(",
    "emel::text::generator::detail::prefill_chunk4_q8_gemm_backend_ready(",
    "emel::text::generator::detail::prefill_chunk8_q8_k_backend_ready(",
    "emel::text::generator::detail::prefill_chunk4_backend_ready<",
    "emel::text::generator::detail::prefill_chunk8_backend_ready<",
    "emel::text::generator::detail::flash_attention_supported(",
    "emel::text::generator::detail::block_uses_attention(",
  };
  for (const auto pattern : forbidden_parent_patterns) {
    CAPTURE(pattern);
    CHECK(parent_guards.find(pattern) == std::string::npos);
    CHECK(prefill_guards.find(pattern) == std::string::npos);
  }
}

TEST_CASE("generator_scalar_kernel_route_choice_stays_in_state_machines") {
  const auto read_source = [](const std::filesystem::path & path) {
    std::ifstream stream(path);
    REQUIRE(stream.good());
    return std::string(
        (std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  };

  const std::string detail_source =
      read_source(repo_root() / "src" / "emel" / "text" / "generator" / "detail.hpp");
  CHECK(detail_source.find("block_uses_attention(") == std::string::npos);
  CHECK(detail_source.find("prefill_chunk4_q8_gemm_backend_ready(") ==
        std::string::npos);
  CHECK(detail_source.find("prefill_chunk8_q8_k_backend_ready(") ==
        std::string::npos);
  CHECK(detail_source.find("prefill_chunk4_backend_ready<") == std::string::npos);
  CHECK(detail_source.find("chunk4_matmul_backend_ready(") == std::string::npos);
  CHECK(detail_source.find("run_kernel_mode(") == std::string::npos);
  CHECK(detail_source.find("run_kernel_mode_preselected_argmax(") == std::string::npos);
  CHECK(detail_source.find("plan->kind == step_kind::prefill ?") == std::string::npos);
  CHECK(detail_source.find("phase_lifecycle(") == std::string::npos);

  const auto guarded_validate_sig = detail_source.find("validate_guarded_compute(");
  REQUIRE(guarded_validate_sig != std::string::npos);
  const auto guarded_validate_start = detail_source.find('{', guarded_validate_sig);
  REQUIRE(guarded_validate_start != std::string::npos);
  const auto guarded_validate_end =
      detail_source.find("validate_guarded_preselected_argmax(", guarded_validate_start);
  REQUIRE(guarded_validate_end != std::string::npos);
  const auto guarded_validate_callbacks =
      detail_source.substr(guarded_validate_start + 1,
                           guarded_validate_end - guarded_validate_start - 1);
  const std::array<std::string_view, 10> forbidden_validate_callback_patterns{
    "request_plan(",
    "check_backend(",
    "expected_outputs",
    "selected_token_out == nullptr",
    "selected_score_out == nullptr",
    "positions_count !=",
    "token_count <=",
    "*err_out =",
    "k_error_invalid",
    "return false",
  };
  for (const auto pattern : forbidden_validate_callback_patterns) {
    CAPTURE(pattern);
    CHECK(guarded_validate_callbacks.find(pattern) == std::string::npos);
  }

  const auto guarded_bind_sig = detail_source.find("bind_guarded_inputs(");
  REQUIRE(guarded_bind_sig != std::string::npos);
  const auto guarded_bind_start = detail_source.find('{', guarded_bind_sig);
  REQUIRE(guarded_bind_start != std::string::npos);
  const auto guarded_bind_end =
      detail_source.find("template <emel::text::generator::attention_mode",
                         guarded_bind_start);
  REQUIRE(guarded_bind_end != std::string::npos);
  const auto guarded_bind_callback =
      detail_source.substr(guarded_bind_start + 1,
                           guarded_bind_end - guarded_bind_start - 1);
  const std::array<std::string_view, 7> forbidden_bind_callback_patterns{
    "request_plan(",
    "check_backend(",
    "== nullptr",
    "expected_outputs",
    "*err_out =",
    "k_error_invalid",
    "return false",
  };
  for (const auto pattern : forbidden_bind_callback_patterns) {
    CAPTURE(pattern);
    CHECK(guarded_bind_callback.find(pattern) == std::string::npos);
  }

  const auto guarded_extract_sig = detail_source.find("extract_guarded_outputs(");
  REQUIRE(guarded_extract_sig != std::string::npos);
  const auto guarded_extract_start = detail_source.find('{', guarded_extract_sig);
  REQUIRE(guarded_extract_start != std::string::npos);
  const auto guarded_extract_end =
      detail_source.find("extract_guarded_preselected_argmax(", guarded_extract_start);
  REQUIRE(guarded_extract_end != std::string::npos);
  const auto guarded_extract_callbacks =
      detail_source.substr(guarded_extract_start + 1,
                           guarded_extract_end - guarded_extract_start - 1);
  const std::array<std::string_view, 8> forbidden_extract_callback_patterns{
    "check_backend(",
    "== nullptr",
    "selected_token_out == nullptr",
    "selected_score_out == nullptr",
    "*err_out =",
    "k_error_invalid",
    "return false",
    "request_plan(",
  };
  for (const auto pattern : forbidden_extract_callback_patterns) {
    CAPTURE(pattern);
    CHECK(guarded_extract_callbacks.find(pattern) == std::string::npos);
  }

  const auto audited_start = detail_source.find("run_kernel_scalar_mode(");
  REQUIRE(audited_start != std::string::npos);
  const auto audited_end =
      detail_source.find("extract_guarded_outputs(", audited_start);
  REQUIRE(audited_end != std::string::npos);
  const auto audited_wrappers = detail_source.substr(audited_start, audited_end - audited_start);
  const std::array<std::string_view, 10> forbidden_wrapper_patterns{
    "request_plan(",
    "plan->kind",
    "check_backend(",
    "bound_ready",
    "selected_token_out == nullptr",
    "selected_score_out == nullptr",
    "prefill_chunk4_backend_ready",
    "prefill_chunk8_q8_k_backend_ready",
    "*err_out =",
    "k_error_invalid",
  };
  for (const auto pattern : forbidden_wrapper_patterns) {
    CAPTURE(pattern);
    CHECK(audited_wrappers.find(pattern) == std::string::npos);
  }

  auto matmul_start = detail_source.find("inline bool matmul_vector(native_backend");
  REQUIRE(matmul_start != std::string::npos);
  matmul_start = detail_source.find("inline bool matmul_vector(native_backend", matmul_start + 1u);
  REQUIRE(matmul_start != std::string::npos);
  const auto matmul_end = detail_source.find("inline bool matmul_vector_argmax", matmul_start);
  REQUIRE(matmul_end != std::string::npos);
  const auto matmul_body = detail_source.substr(matmul_start, matmul_end - matmul_start);
  CHECK(matmul_body.find("packed_q8_0_input_path_supported") == std::string::npos);
  CHECK(matmul_body.find("q8_input_path_supported") == std::string::npos);

  const auto native_matmul_start =
      detail_source.find("inline bool matmul_vector_native_quantized(");
  REQUIRE(native_matmul_start != std::string::npos);
  const auto native_matmul_end =
      detail_source.find("inline bool matmul_vector(", native_matmul_start);
  REQUIRE(native_matmul_end != std::string::npos);
  const auto native_matmul_body =
      detail_source.substr(native_matmul_start, native_matmul_end - native_matmul_start);
  CHECK(native_matmul_body.find("packed_q8_0_input_path_supported") == std::string::npos);
  CHECK(native_matmul_body.find("q8_input_path_supported") == std::string::npos);
  CHECK(native_matmul_body.find("matmul_vector_packed_q8_0(") == std::string::npos);
  CHECK(native_matmul_body.find("matmul_vector_q8_k(") == std::string::npos);

  const std::string parent_actions =
      read_source(repo_root() / "src" / "emel" / "text" / "generator" / "actions.hpp");
  CHECK(parent_actions.find("detail::validate,") == std::string::npos);
  CHECK(parent_actions.find("detail::validate_preselected_argmax") == std::string::npos);
  CHECK(parent_actions.find("detail::bind_inputs") == std::string::npos);
  CHECK(parent_actions.find("detail::extract_outputs") == std::string::npos);
  CHECK(parent_actions.find("detail::extract_preselected_argmax") == std::string::npos);
  CHECK(parent_actions.find("detail::validate_guarded_compute") != std::string::npos);
  CHECK(parent_actions.find("detail::validate_guarded_preselected_argmax") !=
        std::string::npos);
  CHECK(parent_actions.find("detail::bind_guarded_inputs") != std::string::npos);
  CHECK(parent_actions.find("detail::extract_guarded_outputs") != std::string::npos);
  CHECK(parent_actions.find("detail::extract_guarded_preselected_argmax") !=
        std::string::npos);
  CHECK(parent_actions.find("io.selected_token_out = &ev.ctx.selected_token") !=
        std::string::npos);
  CHECK(parent_actions.find("io.selected_score_out = &ev.ctx.selected_score") !=
        std::string::npos);
  CHECK(parent_actions.find("detail::run_kernel_flash>") == std::string::npos);
  CHECK(parent_actions.find("detail::run_kernel_nonflash>") == std::string::npos);
  CHECK(parent_actions.find("detail::run_kernel_flash_preselected_argmax>") ==
        std::string::npos);
  CHECK(parent_actions.find("detail::run_kernel_nonflash_preselected_argmax>") ==
        std::string::npos);

  const std::string parent_sm =
      read_source(repo_root() / "src" / "emel" / "text" / "generator" / "sm.hpp");
  CHECK(parent_sm.find("guard::guard_decode_compute_invalid_request{}") !=
        std::string::npos);
  CHECK(parent_sm.find("guard::guard_decode_compute_backend_unavailable{}") !=
        std::string::npos);
  CHECK(parent_sm.find("guard::guard_decode_materialized_scalar_kernel_ready{}") !=
        std::string::npos);
  CHECK(parent_sm.find("guard::guard_decode_preselected_argmax_kernel_ready{}") !=
        std::string::npos);

  const std::string prefill_actions =
      read_source(repo_root() / "src" / "emel" / "text" / "generator" / "prefill" /
                  "actions.hpp");
  CHECK(prefill_actions.find("detail::run_kernel_flash>") == std::string::npos);
  CHECK(prefill_actions.find("detail::run_kernel_nonflash>") == std::string::npos);
  CHECK(prefill_actions.find("detail::run_kernel_flash_preselected_argmax>") ==
        std::string::npos);
  CHECK(prefill_actions.find("detail::run_kernel_nonflash_preselected_argmax>") ==
        std::string::npos);

  const std::string prefill_sm =
      read_source(repo_root() / "src" / "emel" / "text" / "generator" / "prefill" /
                  "sm.hpp");
  CHECK(prefill_sm.find("guard::guard_compute_invalid_request{}") != std::string::npos);
  CHECK(prefill_sm.find("guard::guard_compute_backend_unavailable{}") != std::string::npos);
  CHECK(prefill_sm.find("guard::guard_materialized_logits_with_scalar_kernel_ready{}") !=
        std::string::npos);
  CHECK(prefill_sm.find("guard::guard_preselected_argmax_with_scalar_kernel_ready{}") !=
        std::string::npos);
}

TEST_CASE("generator_layer_route_actor_keeps_residual_choice_explicit") {
  const auto read_source = [](const std::filesystem::path & path) {
    std::ifstream stream(path);
    REQUIRE(stream.good());
    return std::string(
        (std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  };

  const auto layer_root =
      repo_root() / "src" / "emel" / "text" / "generator" / "layer";
  const std::string layer_sm = read_source(layer_root / "sm.hpp");
  const std::string layer_actions = read_source(layer_root / "actions.hpp");
  const std::string layer_events = read_source(layer_root / "events.hpp");
  const std::string detail_source =
      read_source(repo_root() / "src" / "emel" / "text" / "generator" /
                  "detail.hpp");

  CHECK(layer_sm.find("action::effect_prepare_scalar") != std::string::npos);
  CHECK(layer_sm.find("action::effect_normalize_scalar") != std::string::npos);
  CHECK(layer_sm.find("action::effect_normalize_chunk4") != std::string::npos);
  CHECK(layer_sm.find("action::effect_normalize_chunk8") != std::string::npos);
  CHECK(layer_sm.find("guard::guard_stream_ready{}") != std::string::npos);
  CHECK(layer_sm.find("guard::guard_stream_failed{}") != std::string::npos);
  CHECK(layer_sm.find("guard::guard_normalized_ok{}") != std::string::npos);
  CHECK(layer_sm.find("guard::guard_normalized_failed{}") !=
        std::string::npos);
  CHECK(layer_sm.find("guard::guard_scalar_normalized_attention_route<") !=
        std::string::npos);
  CHECK(layer_sm.find("guard::guard_scalar_normalized_shortconv_route{}") !=
        std::string::npos);
  CHECK(layer_sm.find("guard::guard_chunk4_normalized_attention_route<") !=
        std::string::npos);
  CHECK(layer_sm.find("guard::guard_chunk4_normalized_shortconv_route{}") !=
        std::string::npos);
  CHECK(layer_sm.find("guard::guard_chunk8_normalized_attention_route<") !=
        std::string::npos);
  CHECK(layer_sm.find("guard::guard_chunk8_normalized_shortconv_route{}") !=
        std::string::npos);
  CHECK(layer_sm.find("sml::state<state_input_ready> <= *sml::state<state_idle>") !=
        std::string::npos);
  CHECK(layer_sm.find("sml::state<state_normalized> <= *sml::state<state_idle>") !=
        std::string::npos);
  CHECK(layer_sm.find("+ sml::event<event::scalar_run>") != std::string::npos);
  CHECK(layer_sm.find("+ sml::event<event::chunk4_run>") != std::string::npos);
  CHECK(layer_sm.find("+ sml::event<event::chunk8_run>") != std::string::npos);
  CHECK(layer_sm.find("sml::completion<event::scalar_run>") != std::string::npos);
  CHECK(layer_sm.find("sml::completion<event::chunk4_run>") != std::string::npos);
  CHECK(layer_sm.find("sml::completion<event::chunk8_run>") != std::string::npos);
  CHECK(layer_sm.find("guard::guard_residual_ok{}") != std::string::npos);
  CHECK(layer_sm.find("guard::guard_residual_failed{}") != std::string::npos);
  CHECK(layer_sm.find("guard::guard_feed_forward_ok{}") != std::string::npos);
  CHECK(layer_sm.find("guard::guard_feed_forward_failed{}") !=
        std::string::npos);
  CHECK(layer_sm.find("action::effect_mark_succeeded") != std::string::npos);
  CHECK(layer_sm.find("action::effect_mark_failed") != std::string::npos);

  CHECK(layer_actions.find(" if (") == std::string::npos);
  CHECK(layer_actions.find(" switch") == std::string::npos);
  CHECK(layer_actions.find(" ? ") == std::string::npos);
  CHECK(layer_actions.find("&&") == std::string::npos);
  CHECK(layer_actions.find("ev.ok") == std::string::npos);
  CHECK(layer_actions.find("detail::block_uses_attention") == std::string::npos);

  CHECK(layer_events.find("emel::model::generation_residual_route") !=
        std::string::npos);
  CHECK(layer_events.find("emel::model::llama") == std::string::npos);
  CHECK(layer_events.find("bool &ok") == std::string::npos);
  CHECK(layer_events.find("mutable bool residual_ok") != std::string::npos);
  CHECK(layer_events.find("mutable bool feed_forward_ok") != std::string::npos);
  CHECK(layer_events.find("mutable bool succeeded") != std::string::npos);

  CHECK(detail_source.find("actor.process_event(ev) &&") == std::string::npos);
  CHECK(detail_source.find("layer::scalar_sm<") == std::string::npos);
  CHECK(detail_source.find("layer::chunk4_sm<") == std::string::npos);
  CHECK(detail_source.find("layer::chunk8_sm<") == std::string::npos);
  CHECK(layer_sm.find("process_scalar<") != std::string::npos);
  CHECK(layer_sm.find("process_chunk4<") != std::string::npos);
  CHECK(layer_sm.find("process_chunk8<") != std::string::npos);
  CHECK(layer_actions.find("run_layer(") == std::string::npos);
  CHECK(layer_actions.find("run_layer_chunk4(") == std::string::npos);
  CHECK(layer_actions.find("run_layer_chunk8_q8_k(") == std::string::npos);
  CHECK(detail_source.find("layer::process_scalar<") == std::string::npos);
  CHECK(detail_source.find("layer::process_chunk4<") == std::string::npos);
  CHECK(detail_source.find("layer::process_chunk8<") == std::string::npos);
}

TEST_CASE("docs_detail_shortens_lambda_type_names_for_mermaid") {
  using emel::docs::detail::shorten_type_name;

  CHECK(shorten_type_name("lambda at /tmp/path/my_action.cpp:42:7>") == "lambda_my_action_42_7");
  CHECK(shorten_type_name("lambda at my_action.cpp:42>") == "lambda_my_action_42");
  CHECK(shorten_type_name("lambda at my_action.cpp>") == "lambda_my_action");
}

TEST_CASE("docs_detail_table_event_name_supports_non_completion_event") {
  const auto event_name =
      emel::docs::detail::table_event_name<emel::text::generator::event::generate_run>();
  CHECK(event_name == "generate_run");
}
