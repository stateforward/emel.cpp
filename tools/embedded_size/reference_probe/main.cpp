#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "llama-context.h"
#include "llama-memory.h"
#include "llama-vocab.h"
#include "llama.h"

namespace {

using llama_model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using llama_context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

constexpr std::string_view k_prompt = "hello";
constexpr int32_t k_max_tokens = 1;
constexpr size_t k_output_capacity = 4096u;
constexpr std::array<std::string_view, 9> k_supported_qwen_primary_template_markers = {
    "<|im_start|>",
    "<|im_end|>",
    "messages[0].role == 'system'",
    "message.role == \"user\"",
    "message.role == \"assistant\"",
    "add_generation_prompt",
    "enable_thinking",
    "<tool_call>",
    "tool_response",
};

struct generation_result {
  std::array<char, k_output_capacity> output = {};
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

bool template_matches_supported_qwen_contract(const std::string_view primary_template) noexcept {
  for (const std::string_view marker : k_supported_qwen_primary_template_markers) {
    if (primary_template.find(marker) == std::string_view::npos) {
      return false;
    }
  }
  return true;
}

llama_model_ptr load_reference_model(const char * model_path) {
  llama_model_params params = llama_model_default_params();
  params.n_gpu_layers = 0;
  params.check_tensors = false;
  return llama_model_ptr{llama_model_load_from_file(model_path, params), llama_model_free};
}

llama_context_ptr make_reference_context(llama_model * model) {
  llama_context_params params = llama_context_default_params();
  params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  params.n_ctx = 0;
  params.n_batch = 512;
  params.n_ubatch = 512;
  params.n_seq_max = 1;
  params.n_threads = 1;
  params.n_threads_batch = 1;
  params.embeddings = false;
  return llama_context_ptr{model != nullptr ? llama_init_from_model(model, params) : nullptr,
                           llama_free};
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

bool format_reference_single_user_prompt(const llama_model * model,
                                         const std::string_view text,
                                         std::string & output) {
  const char * primary_template = model != nullptr ? llama_model_chat_template(model, nullptr)
                                                   : nullptr;
  if (primary_template == nullptr || primary_template[0] == '\0' ||
      !template_matches_supported_qwen_contract(primary_template)) {
    output.clear();
    return false;
  }

  std::string prompt_text{text};
  const llama_chat_message message = {
      "user",
      prompt_text.c_str(),
  };

  int32_t output_capacity =
      std::max<int32_t>(64, static_cast<int32_t>(text.size()) * 4 + 64);
  for (;;) {
    output.resize(static_cast<size_t>(output_capacity));
    const int32_t output_length = llama_chat_apply_template(primary_template,
                                                            &message,
                                                            1u,
                                                            true,
                                                            output.data(),
                                                            output_capacity);
    if (output_length < 0) {
      output.clear();
      return false;
    }
    if (output_length <= output_capacity) {
      output.resize(static_cast<size_t>(output_length));
      return true;
    }
    output_capacity = output_length;
  }
}

llama_token select_argmax_token_from_logits(const float * logits, const int32_t vocab_size) {
  int32_t best_index = 0;
  float best_score = logits[0];
  for (int32_t index = 1; index < vocab_size; ++index) {
    if (logits[index] > best_score) {
      best_score = logits[index];
      best_index = index;
    }
  }
  return static_cast<llama_token>(best_index);
}

bool tokenize_reference_prompt(const llama_model * model,
                               const llama_vocab * vocab,
                               std::vector<llama_token> & tokens_out) {
  if (model == nullptr || vocab == nullptr) {
    return false;
  }

  std::string formatted_prompt = {};
  if (!format_reference_single_user_prompt(model, k_prompt, formatted_prompt)) {
    return false;
  }

  int32_t token_capacity =
      std::max<int32_t>(8, static_cast<int32_t>(formatted_prompt.size()) + 8);
  tokens_out.resize(static_cast<size_t>(token_capacity));
  int32_t token_count = llama_tokenize(vocab,
                                       formatted_prompt.data(),
                                       static_cast<int32_t>(formatted_prompt.size()),
                                       tokens_out.data(),
                                       token_capacity,
                                       false,
                                       false);
  if (token_count < 0) {
    token_capacity = -token_count;
    tokens_out.resize(static_cast<size_t>(token_capacity));
    token_count = llama_tokenize(vocab,
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

bool append_reference_piece(const llama_vocab * vocab,
                            const llama_token token,
                            generation_result & result_out) {
  if (vocab == nullptr || result_out.output_length >= result_out.output.size()) {
    return false;
  }

  if (llama_vocab_is_control(vocab, token) || llama_vocab_is_eog(vocab, token)) {
    return true;
  }

  const char * piece = llama_vocab_get_text(vocab, token);
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

bool run_reference_generate(llama_model * model,
                            llama_context * ctx,
                            const llama_vocab * vocab,
                            const int32_t vocab_size,
                            generation_result & result_out) {
  if (model == nullptr || ctx == nullptr || vocab == nullptr || vocab_size <= 0) {
    return false;
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_reference_prompt(model, vocab, prompt_tokens) || !reset_reference_context(ctx)) {
    return false;
  }

  result_out = {};
  llama_batch prompt_batch =
      llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
  if (llama_decode(ctx, prompt_batch) != 0) {
    return false;
  }

  for (int32_t step = 0; step < k_max_tokens; ++step) {
    float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
      return false;
    }

    const llama_token selected = select_argmax_token_from_logits(logits, vocab_size);
    result_out.tokens_generated += 1;
    if (!append_reference_piece(vocab, selected, result_out)) {
      return false;
    }
    if (llama_vocab_is_eog(vocab, selected)) {
      break;
    }

    llama_token next_token = selected;
    llama_batch decode_batch = llama_batch_get_one(&next_token, 1);
    if (llama_decode(ctx, decode_batch) != 0) {
      return false;
    }
  }

  return result_out.tokens_generated > 0;
}

}  // namespace

int main(int argc, char ** argv) {
  if (argc != 2) {
    return 0;
  }

  llama_backend_init();

  const llama_model_ptr model = load_reference_model(argv[1]);
  const llama_vocab * vocab = model != nullptr ? llama_model_get_vocab(model.get()) : nullptr;
  const int32_t vocab_size = vocab != nullptr ? llama_vocab_n_tokens(vocab) : 0;
  const llama_context_ptr ctx = make_reference_context(model.get());

  generation_result result{};
  const bool ok = model != nullptr && ctx != nullptr && vocab != nullptr && vocab_size > 0 &&
      run_reference_generate(model.get(), ctx.get(), vocab, vocab_size, result);

  llama_backend_free();
  if (!ok) {
    return 1;
  }

  volatile size_t sink = result.output_length + static_cast<size_t>(result.tokens_generated);
  return sink == 0u ? 1 : 0;
}
