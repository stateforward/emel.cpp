#include "emel/model/detail.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <span>

#include "emel/model/gemma4/detail.hpp"
#include "emel/model/lfm2/detail.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/qwen3/detail.hpp"

namespace emel::model::detail {

bool load_hparams_from_gguf(const kv_binding & binding,
                            emel::model::data & model_out) noexcept {
  model_out.params = {};

  const auto * architecture_entry = find_kv_entry(binding, "general.architecture");
  if (architecture_entry == nullptr) {
    return false;
  }

  std::string_view architecture_name = {};
  if (!decode_string_value(binding, *architecture_entry, architecture_name) ||
      architecture_name.size() >= model_out.architecture_name.size()) {
    return false;
  }
  copy_name(model_out.architecture_name, architecture_name);

  const hparam_loader loader{binding};
  const bool ok =
      (architecture_name == "llama" && emel::model::llama::detail::load_hparams(loader, model_out)) ||
      (architecture_name == "qwen3" && emel::model::qwen3::detail::load_hparams(loader, model_out)) ||
      (architecture_name == "lfm2" && emel::model::lfm2::detail::load_hparams(loader, model_out)) ||
      (architecture_name == "gemma4" &&
       emel::model::gemma4::detail::load_hparams(loader, model_out));
  if (!ok) {
    return false;
  }

  const auto * tokens_entry =
      find_kv_entry_any(binding, {"tokenizer.tokens", "tokenizer.ggml.tokens"});
  if (tokens_entry != nullptr) {
    uint32_t token_count = 0u;
    if (!decode_string_array_count(binding, *tokens_entry, token_count) ||
        token_count > static_cast<uint32_t>(emel::model::data::k_max_vocab_tokens)) {
      return false;
    }
    model_out.vocab_data.n_tokens = token_count;
    if (model_out.params.n_vocab == 0) {
      model_out.params.n_vocab = static_cast<int32_t>(token_count);
    }
  }

  return true;
}

emel::model::data::tokenizer_model tokenizer_model_from_name(const std::string_view name) noexcept {
  using tokenizer_model = emel::model::data::tokenizer_model;

  if (name == "none" || name == "no_vocab") {
    return tokenizer_model::NONE;
  }
  if (name == "llama") {
    return tokenizer_model::SPM;
  }
  if (name == "gemma4") {
    return tokenizer_model::SPM;
  }
  if (name == "gpt2") {
    return tokenizer_model::BPE;
  }
  if (name == "bert") {
    return tokenizer_model::WPM;
  }
  if (name == "t5") {
    return tokenizer_model::UGM;
  }
  if (name == "rwkv") {
    return tokenizer_model::RWKV;
  }
  if (name == "plamo2") {
    return tokenizer_model::PLAMO2;
  }
  return tokenizer_model::UNKNOWN;
}

emel::model::data::tokenizer_pre tokenizer_pre_profile_from_name(
    const std::string_view name) noexcept {
  using tokenizer_pre = emel::model::data::tokenizer_pre;

  if (name.empty() || name == "default") {
    return tokenizer_pre::DEFAULT;
  }
  if (name == "llama3" || name == "llama-v3" || name == "llama-bpe" ||
      name == "falcon3" || name == "falcon-h1" || name == "pixtral" ||
      name == "midm-2.0" || name == "lfm2" || name == "jina-v5-nano") {
    return tokenizer_pre::LLAMA3;
  }
  if (name == "jais2") {
    return tokenizer_pre::JAIS2;
  }
  if (name == "dbrx") {
    return tokenizer_pre::DBRX;
  }
  if (name == "smaug") {
    return tokenizer_pre::SMAUG;
  }
  if (name == "deepseek-llm") {
    return tokenizer_pre::DEEPSEEK_LLM;
  }
  if (name == "deepseek-coder") {
    return tokenizer_pre::DEEPSEEK_CODER;
  }
  if (name == "deepseek-v3") {
    return tokenizer_pre::DEEPSEEK3_LLM;
  }
  if (name == "youtu") {
    return tokenizer_pre::YOUTU;
  }
  if (name == "falcon") {
    return tokenizer_pre::FALCON;
  }
  if (name == "mpt") {
    return tokenizer_pre::MPT;
  }
  if (name == "starcoder") {
    return tokenizer_pre::STARCODER;
  }
  if (name == "gpt2" || name == "gpt-2") {
    return tokenizer_pre::GPT2;
  }
  if (name == "jais") {
    return tokenizer_pre::JAIS;
  }
  if (name == "refact") {
    return tokenizer_pre::REFACT;
  }
  if (name == "command-r") {
    return tokenizer_pre::COMMAND_R;
  }
  if (name == "qwen2") {
    return tokenizer_pre::QWEN2;
  }
  if (name == "qwen2.5" || name == "qwen35") {
    return tokenizer_pre::QWEN35;
  }
  if (name == "stablelm2") {
    return tokenizer_pre::STABLELM2;
  }
  if (name == "olmo") {
    return tokenizer_pre::OLMO;
  }
  if (name == "poro") {
    return tokenizer_pre::PORO;
  }
  if (name == "chatglm4") {
    return tokenizer_pre::CHATGLM4;
  }
  if (name == "viking") {
    return tokenizer_pre::VIKING;
  }
  if (name == "tekken") {
    return tokenizer_pre::TEKKEN;
  }
  if (name == "smollm") {
    return tokenizer_pre::SMOLLM;
  }
  if (name == "codeshell") {
    return tokenizer_pre::CODESHELL;
  }
  if (name == "bloom") {
    return tokenizer_pre::BLOOM;
  }
  if (name == "gpt3-finnish") {
    return tokenizer_pre::GPT3_FINNISH;
  }
  if (name == "exaone") {
    return tokenizer_pre::EXAONE;
  }
  if (name == "exaone4") {
    return tokenizer_pre::EXAONE4;
  }
  if (name == "exaone-moe") {
    return tokenizer_pre::EXAONE_MOE;
  }
  if (name == "chameleon") {
    return tokenizer_pre::CHAMELEON;
  }
  if (name == "minerva") {
    return tokenizer_pre::MINERVA;
  }
  if (name == "megrez") {
    return tokenizer_pre::MEGREZ;
  }
  if (name == "gpt4o" || name == "gpt-4o") {
    return tokenizer_pre::GPT4O;
  }
  if (name == "tiny-aya") {
    return tokenizer_pre::TINY_AYA;
  }
  if (name == "superbpe") {
    return tokenizer_pre::SUPERBPE;
  }
  if (name == "trillion") {
    return tokenizer_pre::TRILLION;
  }
  if (name == "granite-docling") {
    return tokenizer_pre::GRANITE_DOCLING;
  }
  if (name == "bailingmoe") {
    return tokenizer_pre::BAILINGMOE;
  }
  if (name == "seed-coder") {
    return tokenizer_pre::SEED_CODER;
  }
  if (name == "hunyuan") {
    return tokenizer_pre::HUNYUAN;
  }
  if (name == "hunyuan-dense") {
    return tokenizer_pre::HUNYUAN_DENSE;
  }
  if (name == "joyai-llm") {
    return tokenizer_pre::JOYAI_LLM;
  }
  if (name == "kimi-k2") {
    return tokenizer_pre::KIMI_K2;
  }
  if (name == "grok-2") {
    return tokenizer_pre::GROK_2;
  }
  if (name == "afmoe") {
    return tokenizer_pre::AFMOE;
  }
  if (name == "minimax-m2") {
    return tokenizer_pre::MINIMAX_M2;
  }
  if (name == "solar-open") {
    return tokenizer_pre::SOLAR_OPEN;
  }
  return tokenizer_pre::UNKNOWN;
}

void apply_tokenizer_model_defaults(const std::string_view name,
                                    emel::model::data::vocab & vocab) noexcept {
  if (name == "llama") {
    vocab.bos_id = 1;
    vocab.eos_id = 2;
    vocab.unk_id = 0;
    vocab.add_bos = true;
    vocab.add_space_prefix = true;
    vocab.escape_whitespaces = true;
    return;
  }

  if (name == "bert") {
    vocab.bos_id = 101;
    vocab.unk_id = 100;
    vocab.sep_id = 102;
    vocab.pad_id = 0;
    vocab.mask_id = 103;
    vocab.add_bos = true;
    vocab.add_sep = true;
    return;
  }

  if (name == "gpt2") {
    vocab.bos_id = 11;
    vocab.eos_id = 11;
    return;
  }

  if (name == "t5") {
    vocab.eos_id = 1;
    vocab.unk_id = 2;
    vocab.pad_id = 0;
    return;
  }

  if (name == "plamo2") {
    vocab.bos_id = 1;
    vocab.eos_id = 2;
    vocab.unk_id = 0;
    vocab.pad_id = 3;
  }
}

void apply_tokenizer_pre_defaults(const std::string_view name,
                                  emel::model::data::vocab & vocab) noexcept {
  if (name == "llama3" || name == "llama-v3" || name == "llama-bpe" ||
      name == "falcon3" || name == "falcon-h1" || name == "pixtral" ||
      name == "midm-2.0" || name == "lfm2" || name == "jina-v5-nano") {
    vocab.ignore_merges = true;
    vocab.add_bos = true;
    return;
  }

  if (name == "youtu") {
    vocab.ignore_merges = true;
  }
}

void mark_special_token_type(emel::model::data::vocab & vocab,
                             const int32_t token_id,
                             const int32_t token_type) noexcept {
  if (token_id < 0 || static_cast<uint32_t>(token_id) >= vocab.n_tokens) {
    return;
  }

  auto & entry = vocab.entries[static_cast<size_t>(token_id)];
  if (entry.type == k_token_type_undefined || entry.type == k_token_type_normal) {
    entry.type = token_type;
  }
}

bool load_vocab_from_gguf(const kv_binding & binding,
                          emel::model::data::vocab & vocab_out) noexcept {
  const auto fail = [](const char * stage) noexcept {
    if (std::getenv("EMEL_DEBUG_GGUF_VOCAB") != nullptr) {
      std::fprintf(stderr, "load_vocab_from_gguf failed at %s\n", stage);
    }
    return false;
  };

  vocab_out = {};

  const auto * model_entry =
      find_kv_entry_any(binding, {"tokenizer.model", "tokenizer.ggml.model"});
  if (model_entry == nullptr) {
    return fail("missing_tokenizer_model");
  }

  std::string_view tokenizer_model_name = {};
  if (!decode_string_value(binding, *model_entry, tokenizer_model_name)) {
    return fail("decode_tokenizer_model");
  }

  std::string_view tokenizer_pre_name = {};
  if (const auto * pre_entry =
          find_kv_entry_any(binding, {"tokenizer.pre", "tokenizer.ggml.pre"});
      pre_entry != nullptr &&
      !decode_string_value(binding, *pre_entry, tokenizer_pre_name)) {
    return fail("decode_tokenizer_pre");
  }

  vocab_out.tokenizer_model_id = tokenizer_model_from_name(tokenizer_model_name);
  vocab_out.tokenizer_pre_id = tokenizer_pre_profile_from_name(tokenizer_pre_name);
  copy_name(vocab_out.tokenizer_model_name, tokenizer_model_name);
  copy_name(vocab_out.tokenizer_pre_name, tokenizer_pre_name);
  apply_tokenizer_model_defaults(tokenizer_model_name, vocab_out);
  apply_tokenizer_pre_defaults(tokenizer_pre_name, vocab_out);

  if (const auto * type_count_entry = find_kv_entry_any(
          binding,
          {"tokenizer.token_type_count", "tokenizer.ggml.token_type_count"});
      type_count_entry != nullptr) {
    uint64_t type_count = 0u;
    if (!decode_integer_value(binding, *type_count_entry, type_count) ||
        type_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      return false;
    }
    vocab_out.n_token_types = static_cast<uint32_t>(type_count);
  }

  const auto * tokens_entry =
      find_kv_entry_any(binding, {"tokenizer.tokens", "tokenizer.ggml.tokens"});
  if (tokens_entry == nullptr) {
    return fail("missing_tokens");
  }

  uint32_t token_count = 0u;
  if (!decode_string_array_count(binding, *tokens_entry, token_count) ||
      token_count > emel::model::data::k_max_vocab_tokens) {
    return fail("decode_token_count");
  }

  vocab_out.n_tokens = token_count;

  const auto * scores_entry =
      find_kv_entry_any(binding, {"tokenizer.scores", "tokenizer.ggml.scores"});
  if (scores_entry != nullptr) {
    array_header header = {};
    if (!decode_array_header(binding, *scores_entry, header) || header.count != token_count) {
      return fail("scores_header");
    }
  }

  const auto * types_entry =
      find_kv_entry_any(binding, {"tokenizer.token_type", "tokenizer.ggml.token_type"});
  if (types_entry != nullptr) {
    array_header header = {};
    if (!decode_array_header(binding, *types_entry, header) || header.count != token_count) {
      return fail("types_header");
    }
  }

  uint32_t token_bytes_used = 0u;
  bool token_storage_overflow = false;
  bool token_score_decode_failed = false;
  bool token_type_decode_failed = false;
  if (!visit_string_array_elements(
          binding,
          *tokens_entry,
          [&](const uint32_t token_id, const std::string_view token_text) noexcept {
            if (token_bytes_used + token_text.size() > emel::model::data::k_max_vocab_bytes) {
              token_storage_overflow = true;
              return false;
            }

            if (!token_text.empty()) {
              std::memcpy(vocab_out.token_storage.data() + token_bytes_used,
                          token_text.data(),
                          token_text.size());
            }

            auto & entry = vocab_out.entries[token_id];
            entry.text_offset = token_bytes_used;
            entry.text_length = static_cast<uint32_t>(token_text.size());
            entry.score = 0.0f;
            entry.type = k_token_type_normal;

            if (scores_entry != nullptr &&
                !decode_float_array_element(binding, *scores_entry, token_id, entry.score)) {
              token_score_decode_failed = true;
              return false;
            }

            if (types_entry != nullptr) {
              uint64_t type_value = 0u;
              if (!decode_uint_array_element(binding, *types_entry, token_id, type_value) ||
                  type_value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                token_type_decode_failed = true;
                return false;
              }
              entry.type = static_cast<int32_t>(type_value);
            }

            token_bytes_used += static_cast<uint32_t>(token_text.size());
            return true;
          })) {
    if (token_storage_overflow) {
      return fail("token_storage_overflow");
    }
    if (token_score_decode_failed) {
      return fail("decode_token_score");
    }
    if (token_type_decode_failed) {
      return fail("decode_token_type");
    }
    return fail("decode_token_text");
  }
  vocab_out.token_bytes_used = token_bytes_used;

  if (const auto * merges_entry =
          find_kv_entry_any(binding, {"tokenizer.merges", "tokenizer.ggml.merges"});
      merges_entry != nullptr) {
    uint32_t merge_count = 0u;
    if (!decode_string_array_count(binding, *merges_entry, merge_count) ||
        merge_count > emel::model::data::k_max_merges) {
      return fail("decode_merge_count");
    }

    uint32_t merge_bytes_used = 0u;
    bool merge_storage_overflow = false;
    if (!visit_string_array_elements(
            binding,
            *merges_entry,
            [&](const uint32_t merge_index, const std::string_view merge_text) noexcept {
              if (merge_bytes_used + merge_text.size() > emel::model::data::k_max_merge_bytes) {
                merge_storage_overflow = true;
                return false;
              }

              if (!merge_text.empty()) {
                std::memcpy(vocab_out.merge_storage.data() + merge_bytes_used,
                            merge_text.data(),
                            merge_text.size());
              }

              vocab_out.merge_offsets[merge_index] = merge_bytes_used;
              vocab_out.merge_lengths[merge_index] = static_cast<uint32_t>(merge_text.size());
              merge_bytes_used += static_cast<uint32_t>(merge_text.size());
              return true;
            })) {
      if (merge_storage_overflow) {
        return fail("merge_storage_overflow");
      }
      return fail("decode_merge_text");
    }

    vocab_out.n_merges = merge_count;
    vocab_out.merge_bytes_used = merge_bytes_used;
  }

  if (const auto * charsmap_entry = find_kv_entry_any(
          binding,
          {"tokenizer.precompiled_charsmap", "tokenizer.ggml.precompiled_charsmap"});
      charsmap_entry != nullptr &&
      !decode_byte_array_copy(
          binding,
          *charsmap_entry,
          std::span<uint8_t>{vocab_out.precompiled_charsmap},
          vocab_out.precompiled_charsmap_size)) {
    return fail("decode_precompiled_charsmap");
  }

  const auto assign_i32 = [&](const std::initializer_list<std::string_view> keys,
                              int32_t & field) noexcept {
    const auto * entry = find_kv_entry_any(binding, keys);
    if (entry == nullptr) {
      return true;
    }

    uint64_t value = 0u;
    if (!decode_integer_value(binding, *entry, value) ||
        value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }

    field = static_cast<int32_t>(value);
    return true;
  };

  const auto assign_bool = [&](const std::initializer_list<std::string_view> keys,
                               bool & field) noexcept {
    const auto * entry = find_kv_entry_any(binding, keys);
    if (entry == nullptr) {
      return true;
    }

    bool value = false;
    if (!decode_bool_value(binding, *entry, value)) {
      return false;
    }

    field = value;
    return true;
  };

  if (!assign_i32({"tokenizer.bos_token_id", "tokenizer.ggml.bos_token_id"}, vocab_out.bos_id) ||
      !assign_i32({"tokenizer.eos_token_id", "tokenizer.ggml.eos_token_id"}, vocab_out.eos_id) ||
      !assign_i32({"tokenizer.eot_token_id", "tokenizer.ggml.eot_token_id"}, vocab_out.eot_id) ||
      !assign_i32({"tokenizer.eom_token_id", "tokenizer.ggml.eom_token_id"}, vocab_out.eom_id) ||
      !assign_i32(
          {"tokenizer.unknown_token_id", "tokenizer.ggml.unknown_token_id"},
          vocab_out.unk_id) ||
      !assign_i32(
          {"tokenizer.seperator_token_id", "tokenizer.ggml.seperator_token_id"},
          vocab_out.sep_id) ||
      !assign_i32(
          {"tokenizer.padding_token_id", "tokenizer.ggml.padding_token_id"},
          vocab_out.pad_id) ||
      !assign_i32({"tokenizer.cls_token_id", "tokenizer.ggml.cls_token_id"}, vocab_out.cls_id) ||
      !assign_i32(
          {"tokenizer.mask_token_id", "tokenizer.ggml.mask_token_id"},
          vocab_out.mask_id) ||
      !assign_i32(
          {"tokenizer.prefix_token_id", "tokenizer.ggml.prefix_token_id"},
          vocab_out.prefix_id) ||
      !assign_i32(
          {"tokenizer.suffix_token_id", "tokenizer.ggml.suffix_token_id"},
          vocab_out.suffix_id) ||
      !assign_i32(
          {"tokenizer.middle_token_id", "tokenizer.ggml.middle_token_id"},
          vocab_out.middle_id) ||
      !assign_i32(
          {"tokenizer.fim_pre_token_id", "tokenizer.ggml.fim_pre_token_id"},
          vocab_out.fim_pre_id) ||
      !assign_i32(
          {"tokenizer.fim_suf_token_id", "tokenizer.ggml.fim_suf_token_id"},
          vocab_out.fim_suf_id) ||
      !assign_i32(
          {"tokenizer.fim_mid_token_id", "tokenizer.ggml.fim_mid_token_id"},
          vocab_out.fim_mid_id) ||
      !assign_i32(
          {"tokenizer.fim_pad_token_id", "tokenizer.ggml.fim_pad_token_id"},
          vocab_out.fim_pad_id) ||
      !assign_i32(
          {"tokenizer.fim_rep_token_id", "tokenizer.ggml.fim_rep_token_id"},
          vocab_out.fim_rep_id) ||
      !assign_i32(
          {"tokenizer.fim_sep_token_id", "tokenizer.ggml.fim_sep_token_id"},
          vocab_out.fim_sep_id) ||
      !assign_bool({"tokenizer.add_bos_token", "tokenizer.ggml.add_bos_token"}, vocab_out.add_bos) ||
      !assign_bool({"tokenizer.add_eos_token", "tokenizer.ggml.add_eos_token"}, vocab_out.add_eos) ||
      !assign_bool({"tokenizer.add_sep_token", "tokenizer.ggml.add_sep_token"}, vocab_out.add_sep) ||
      !assign_bool(
          {"tokenizer.add_space_prefix", "tokenizer.ggml.add_space_prefix"},
          vocab_out.add_space_prefix) ||
      !assign_bool(
          {"tokenizer.remove_extra_whitespaces", "tokenizer.ggml.remove_extra_whitespaces"},
          vocab_out.remove_extra_whitespaces) ||
      !assign_bool(
          {"tokenizer.ignore_merges", "tokenizer.ggml.ignore_merges"},
          vocab_out.ignore_merges) ||
      !assign_bool(
          {"tokenizer.escape_whitespaces", "tokenizer.ggml.escape_whitespaces"},
          vocab_out.escape_whitespaces) ||
      !assign_bool(
          {"tokenizer.treat_whitespace_as_suffix",
           "tokenizer.ggml.treat_whitespace_as_suffix"},
          vocab_out.treat_whitespace_as_suffix)) {
    return fail("assign_special_fields");
  }

  mark_special_token_type(vocab_out, vocab_out.unk_id, k_token_type_unknown);
  mark_special_token_type(vocab_out, vocab_out.bos_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.eos_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.eot_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.eom_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.sep_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.pad_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.cls_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.mask_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.prefix_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.suffix_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.middle_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.fim_pre_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.fim_suf_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.fim_mid_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.fim_pad_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.fim_rep_id, k_token_type_control);
  mark_special_token_type(vocab_out, vocab_out.fim_sep_id, k_token_type_control);

  if (vocab_out.tokenizer_model_id == emel::model::data::tokenizer_model::UNKNOWN) {
    return fail("unknown_tokenizer_model");
  }

  return true;
}

}  // namespace emel::model::detail
