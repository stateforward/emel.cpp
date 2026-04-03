#include "emel/model/detail.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <span>

#include "emel/model/loader/errors.hpp"
#include "emel/text/tokenizer/detail.hpp"
#include "emel/text/tokenizer/preprocessor/detail.hpp"

namespace emel::model::detail {

bool load_hparams_from_gguf(const kv_binding & binding,
                            const emel::model::architectures available_architectures,
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

  const auto * architecture =
      emel::model::resolve_architecture(architecture_name, available_architectures);
  if (architecture == nullptr || architecture->load_hparams == nullptr) {
    return false;
  }

  const hparam_loader loader{binding};
  if (!architecture->load_hparams(loader, model_out)) {
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

bool load_hparams_from_gguf(const kv_binding & binding, emel::model::data & model_out) noexcept {
  return load_hparams_from_gguf(binding, emel::model::default_architecture_span(), model_out);
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

  vocab_out.tokenizer_model_id =
      emel::text::tokenizer::detail::tokenizer_model_from_name(tokenizer_model_name);
  vocab_out.tokenizer_pre_id = emel::text::tokenizer::preprocessor::detail::
      tokenizer_pre_profile_from_name(tokenizer_pre_name);
  copy_name(vocab_out.tokenizer_model_name, tokenizer_model_name);
  copy_name(vocab_out.tokenizer_pre_name, tokenizer_pre_name);
  emel::text::tokenizer::detail::apply_tokenizer_model_defaults(tokenizer_model_name, vocab_out);
  emel::text::tokenizer::preprocessor::detail::apply_tokenizer_pre_defaults(
      tokenizer_pre_name, vocab_out);

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

    int64_t value = 0;
    if (!decode_signed_integer_value(binding, *entry, value) ||
        value < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
        value > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
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
