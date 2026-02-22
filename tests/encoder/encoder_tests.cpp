#include <array>
#include <queue>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <doctest/doctest.h>

#include "emel/encoder/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/bpe/detail.hpp"
#include "emel/encoder/spm/detail.hpp"
#include "emel/encoder/wpm/detail.hpp"
#include "emel/encoder/ugm/detail.hpp"
#include "emel/encoder/rwkv/detail.hpp"
#include "emel/encoder/plamo2/detail.hpp"
#include "emel/encoder/fallback/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/encoder/guards.hpp"
#include "emel/encoder/bpe/sm.hpp"
#include "emel/encoder/spm/sm.hpp"
#include "emel/encoder/wpm/sm.hpp"
#include "emel/encoder/ugm/sm.hpp"
#include "emel/encoder/rwkv/sm.hpp"
#include "emel/encoder/plamo2/sm.hpp"
#include "emel/encoder/fallback/sm.hpp"
#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"

namespace {

emel::model::data::vocab & vocab_storage() {
  static emel::model::data::vocab storage{};
  return storage;
}

size_t sum_offsets(const std::vector<size_t> & offsets) {
  size_t sum = 0;
  for (const size_t value : offsets) {
    sum += value;
  }
  return sum;
}

struct dispatch_recorder {
  int done_count = 0;
  int error_count = 0;
};

bool record_done(void * owner, const emel::encoder::events::encoding_done &) {
  if (owner == nullptr) {
    return false;
  }
  static_cast<dispatch_recorder *>(owner)->done_count += 1;
  return true;
}

bool record_error(void * owner, const emel::encoder::events::encoding_error &) {
  if (owner == nullptr) {
    return false;
  }
  static_cast<dispatch_recorder *>(owner)->error_count += 1;
  return true;
}

struct vocab_builder {
  emel::model::data::vocab * vocab = nullptr;

  vocab_builder() : vocab(&vocab_storage()) {
    std::memset(vocab, 0, sizeof(*vocab));
  }

  void set_model(const char * value) {
    std::memset(vocab->tokenizer_model_name.data(), 0, vocab->tokenizer_model_name.size());
    std::strncpy(vocab->tokenizer_model_name.data(), value, vocab->tokenizer_model_name.size() - 1);
    if (std::strcmp(value, "llama") == 0) {
      vocab->tokenizer_model_id = emel::model::data::tokenizer_model::SPM;
    } else if (std::strcmp(value, "gpt2") == 0) {
      vocab->tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
    } else if (std::strcmp(value, "bert") == 0) {
      vocab->tokenizer_model_id = emel::model::data::tokenizer_model::WPM;
    } else if (std::strcmp(value, "t5") == 0) {
      vocab->tokenizer_model_id = emel::model::data::tokenizer_model::UGM;
    } else if (std::strcmp(value, "rwkv") == 0) {
      vocab->tokenizer_model_id = emel::model::data::tokenizer_model::RWKV;
    } else if (std::strcmp(value, "plamo2") == 0) {
      vocab->tokenizer_model_id = emel::model::data::tokenizer_model::PLAMO2;
    } else if (std::strcmp(value, "none") == 0 || std::strcmp(value, "no_vocab") == 0) {
      vocab->tokenizer_model_id = emel::model::data::tokenizer_model::NONE;
    } else {
      vocab->tokenizer_model_id = emel::model::data::tokenizer_model::UNKNOWN;
    }
  }

  void set_pre(const char * value) {
    std::memset(vocab->tokenizer_pre_name.data(), 0, vocab->tokenizer_pre_name.size());
    std::strncpy(vocab->tokenizer_pre_name.data(), value, vocab->tokenizer_pre_name.size() - 1);
    if (std::strcmp(value, "default") == 0) {
      vocab->tokenizer_pre_id = emel::model::data::tokenizer_pre::DEFAULT;
    } else if (std::strcmp(value, "gpt2") == 0) {
      vocab->tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
    } else if (std::strcmp(value, "llama3") == 0) {
      vocab->tokenizer_pre_id = emel::model::data::tokenizer_pre::LLAMA3;
    } else if (std::strcmp(value, "mpt") == 0) {
      vocab->tokenizer_pre_id = emel::model::data::tokenizer_pre::MPT;
    } else {
      vocab->tokenizer_pre_id = emel::model::data::tokenizer_pre::UNKNOWN;
    }
  }

  int32_t add_token(const char * text, float score, int32_t type) {
    const uint32_t len = static_cast<uint32_t>(std::strlen(text));
    const uint32_t offset = vocab->token_bytes_used;
    std::memcpy(vocab->token_storage.data() + offset, text, len);
    const uint32_t id = vocab->n_tokens;
    vocab->entries[id].text_offset = offset;
    vocab->entries[id].text_length = len;
    vocab->entries[id].score = score;
    vocab->entries[id].type = type;
    vocab->token_bytes_used += len;
    vocab->n_tokens = id + 1;
    return static_cast<int32_t>(id);
  }

  void add_merge(const char * text) {
    const uint32_t len = static_cast<uint32_t>(std::strlen(text));
    const uint32_t offset = vocab->merge_bytes_used;
    std::memcpy(vocab->merge_storage.data() + offset, text, len);
    const uint32_t id = vocab->n_merges;
    vocab->merge_offsets[id] = offset;
    vocab->merge_lengths[id] = len;
    vocab->merge_bytes_used += len;
    vocab->n_merges = id + 1;
  }

  int32_t add_byte_token(uint8_t byte) {
    const std::string token = emel::text::unicode_byte_to_utf8(byte);
    return add_token(token.c_str(), 0.0f, 6);
  }

  int32_t add_plamo2_byte_token(uint8_t byte) {
    char token[7] = {};
    std::snprintf(token, sizeof(token), "<0x%02X>", byte);
    return add_token(token, 0.0f, 6);
  }

  void add_all_byte_tokens() {
    for (int value = 0; value < 256; ++value) {
      add_byte_token(static_cast<uint8_t>(value));
    }
  }

  void add_all_plamo2_byte_tokens() {
    for (int value = 0; value < 256; ++value) {
      add_plamo2_byte_token(static_cast<uint8_t>(value));
    }
  }

  void set_charsmap_a_to_b() {
    uint8_t * data = vocab->precompiled_charsmap.data();
    constexpr uint32_t table_size = 98u;
    const uint32_t blob_size = table_size * static_cast<uint32_t>(sizeof(uint32_t));
    std::memcpy(data, &blob_size, sizeof(blob_size));
    uint32_t * entries = reinterpret_cast<uint32_t *>(data + sizeof(blob_size));
    std::memset(entries, 0, blob_size);

    // root base = 1, so root_base ^ 'a' -> 96
    entries[0] = (1u << 10);
    // node 96: lcheck='a', leaf=1, base=1, so node ^ base -> value node 97
    entries[96] = (1u << 10) | (1u << 8) | static_cast<uint32_t>('a');
    // node 97: value offset = 0 into replacement strings blob
    entries[97] = 0u;

    data[sizeof(blob_size) + blob_size + 0] = 'b';
    data[sizeof(blob_size) + blob_size + 1] = '\0';
    vocab->precompiled_charsmap_size = sizeof(blob_size) + blob_size + 2;
  }
};

}  // namespace

TEST_CASE("encoder_bpe_ignore_merges_prefers_full_token") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t full_id = builder.add_token("hello", 0.5f, 1);
  builder.vocab->ignore_merges = true;

  emel::encoder::bpe::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "hello",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == full_id);
}

TEST_CASE("encoder_wpm_emits_longest_token") {
  vocab_builder builder{};
  builder.set_model("bert");
  const int32_t token_id = builder.add_token("\xE2\x96\x81hello", 0.2f, 1);
  builder.vocab->unk_id = 0;

  emel::encoder::wpm::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "hello",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == token_id);
}

TEST_CASE("encoder_ugm_applies_precompiled_charsmap") {
  vocab_builder builder{};
  builder.set_model("t5");
  const int32_t token_id = builder.add_token("b", 0.1f, 1);
  builder.vocab->unk_id = 2;
  builder.vocab->escape_whitespaces = false;
  builder.vocab->add_space_prefix = false;
  builder.set_charsmap_a_to_b();

  emel::encoder::ugm::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "a",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == token_id);
}

TEST_CASE("encoder_spm_merges_bigram") {
  vocab_builder builder{};
  builder.set_model("llama");
  const int32_t h_id = builder.add_token("h", 0.1f, 1);
  const int32_t i_id = builder.add_token("i", 0.1f, 1);
  const int32_t hi_id = builder.add_token("hi", 0.9f, 1);
  (void)h_id;
  (void)i_id;

  emel::encoder::spm::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "hi",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == hi_id);
}

TEST_CASE("encoder_bpe_merges_ranked_pair") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t he_id = builder.add_token("he", 0.9f, 1);
  builder.add_token("h", 0.1f, 1);
  builder.add_token("e", 0.1f, 1);
  builder.add_merge("h e");

  emel::encoder::bpe::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "he",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == he_id);
}

TEST_CASE("encoder_bpe_byte_fallback") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t byte_id = builder.add_byte_token(static_cast<uint8_t>('!'));

  emel::encoder::bpe::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "!",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == byte_id);
}

TEST_CASE("encoder_ugm_normalization_flags") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.vocab->escape_whitespaces = true;
  builder.vocab->add_space_prefix = true;
  builder.vocab->remove_extra_whitespaces = true;
  builder.vocab->treat_whitespace_as_suffix = false;

  emel::encoder::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;

  std::string_view normalized{};
  CHECK(emel::encoder::ugm::detail::normalize_ugm_into(
    *builder.vocab, ctx, "  hello   world ", normalized));

  CHECK(!normalized.empty());
  CHECK(normalized.find("hello") != std::string_view::npos);
  CHECK(normalized.find("world") != std::string_view::npos);
}

TEST_CASE("unicode_helpers_cover_common_paths") {
  CHECK(emel::text::unicode_len_utf8('a') == 1);
  CHECK(emel::text::unicode_len_utf8(static_cast<char>(0xC2)) == 2);
  CHECK(emel::text::unicode_len_utf8(static_cast<char>(0xE2)) == 3);
  CHECK(emel::text::unicode_len_utf8(static_cast<char>(0xF0)) == 4);

  const std::string euro = emel::text::unicode_cpt_to_utf8(0x20AC);
  const std::vector<uint32_t> cpts = emel::text::unicode_cpts_from_utf8("A" + euro);
  CHECK(cpts.size() == 2);
  CHECK(cpts[0] == static_cast<uint32_t>('A'));
  CHECK(cpts[1] == 0x20AC);

  const emel::text::unicode_cpt_flags flags_a =
    emel::text::unicode_cpt_flags_from_cpt(static_cast<uint32_t>('A'));
  CHECK(flags_a.is_letter);
  CHECK(!flags_a.is_number);
  CHECK(!flags_a.is_whitespace);

  CHECK(emel::text::unicode_cpt_is_han(0x4E00));
}

TEST_CASE("unicode_regex_split_custom_paths") {
  const std::string text = "hello 123!";
  const auto cpts = emel::text::unicode_cpts_from_utf8(text);
  const std::vector<size_t> offsets{cpts.size()};

  const auto gpt2_offsets = emel::text::unicode_regex_split_custom_gpt2(text, offsets);
  CHECK(!gpt2_offsets.empty());
  CHECK(sum_offsets(gpt2_offsets) == cpts.size());

  const auto llama3_offsets = emel::text::unicode_regex_split_custom_llama3(text, offsets);
  CHECK(!llama3_offsets.empty());
  CHECK(sum_offsets(llama3_offsets) == cpts.size());

  const auto kimi_offsets = emel::text::unicode_regex_split_custom_kimi_k2(text, offsets);
  CHECK(!kimi_offsets.empty());
  CHECK(sum_offsets(kimi_offsets) == cpts.size());

  const auto afmoe_offsets = emel::text::unicode_regex_split_custom_afmoe(text, offsets);
  CHECK(!afmoe_offsets.empty());
  CHECK(sum_offsets(afmoe_offsets) == cpts.size());
}

TEST_CASE("unicode_regex_split_collapsed_categories") {
  std::vector<std::string> exprs = {
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
  };
  const std::vector<std::string> parts = emel::text::unicode_regex_split("hello 42", exprs);
  CHECK(!parts.empty());
  CHECK(parts.front().find("hello") != std::string::npos);
}

TEST_CASE("encoder_rejects_invalid_input") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("hello", 0.5f, 1);

  emel::encoder::bpe::sm machine{};

  int32_t token_count = 7;
  int32_t err = EMEL_OK;

  CHECK(!machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "hello",
    .token_ids = nullptr,
    .token_capacity = 0,
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_dispatch_callbacks") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.vocab->ignore_merges = true;
  builder.add_token("hello", 0.5f, 1);

  emel::encoder::bpe::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  dispatch_recorder recorder{};

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "hello",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
    .owner_sm = &recorder,
    .dispatch_done = record_done,
    .dispatch_error = record_error,
  }));

  CHECK(err == EMEL_OK);
  CHECK(recorder.done_count == 1);
  CHECK(recorder.error_count == 0);
}

TEST_CASE("encoder_dispatch_error_on_missing_bytes") {
  vocab_builder builder{};
  builder.set_model("unknown");

  emel::encoder::fallback::sm machine{};

  std::array<int32_t, 1> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  dispatch_recorder recorder{};

  CHECK(!machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "x",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
    .owner_sm = &recorder,
    .dispatch_done = record_done,
    .dispatch_error = record_error,
  }));

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(recorder.done_count == 0);
  CHECK(recorder.error_count == 1);
}

TEST_CASE("encoder_wpm_falls_back_to_unk") {
  vocab_builder builder{};
  builder.set_model("bert");
  const int32_t unk_id = builder.add_token("<unk>", 0.0f, 2);
  builder.vocab->unk_id = unk_id;

  emel::encoder::wpm::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "hello",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == unk_id);
}

TEST_CASE("encoder_fallback_byte_tokens") {
  vocab_builder builder{};
  builder.set_model("unknown");
  const int32_t x_id = builder.add_byte_token(static_cast<uint8_t>('x'));
  const int32_t y_id = builder.add_byte_token(static_cast<uint8_t>('y'));

  emel::encoder::fallback::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "xy",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 2);
  CHECK(tokens[0] == x_id);
  CHECK(tokens[1] == y_id);
}

TEST_CASE("encoder_plamo2_byte_tokens") {
  vocab_builder builder{};
  builder.set_model("plamo2");
  builder.add_token("<unk>", 0.0f, 2);
  builder.add_all_plamo2_byte_tokens();
  const int32_t byte_id = builder.add_plamo2_byte_token(static_cast<uint8_t>('p'));

  emel::encoder::plamo2::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "p",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == byte_id);
}

TEST_CASE("encoder_rwkv_byte_tokens") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  const int32_t byte_id = builder.add_byte_token(static_cast<uint8_t>('r'));

  emel::encoder::rwkv::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::encoder::event::encode{
    .vocab = builder.vocab,
    .text = "r",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == byte_id);
}

TEST_CASE("encoder_unexpected_event_sets_error") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("hello", 0.5f, 1);

  emel::encoder::bpe::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  dispatch_recorder recorder{};

  emel::encoder::event::encode request{
    .vocab = builder.vocab,
    .text = "hello",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
    .owner_sm = &recorder,
    .dispatch_done = record_done,
    .dispatch_error = record_error,
  };

  CHECK(machine.process_event(emel::encoder::events::encoding_done{&request, 0}));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(recorder.error_count == 1);
}

TEST_CASE("encoder_detail_trie_basic") {
  emel::encoder::detail::naive_trie trie{};
  trie.insert("ab", 2, 42);

  const auto * node = trie.traverse('a');
  CHECK(node != nullptr);
  const auto * node_b = node->traverse('b');
  CHECK(node_b != nullptr);
  CHECK(node_b->has_value);
  CHECK(node_b->value == 42);
  CHECK(node->traverse('z') == nullptr);
}

TEST_CASE("encoder_ensure_tables_populates_state") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.set_pre("gpt2");
  builder.add_token("a", 0.2f, 1);
  builder.add_token("b", 0.1f, 4);
  builder.add_token("c", 0.3f, 5);
  builder.add_merge("a b");
  builder.set_charsmap_a_to_b();

  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;

  CHECK(emel::encoder::detail::ensure_tables(ctx));
  CHECK(ctx.tables_ready);
  CHECK(!ctx.token_to_id.empty());
  CHECK(!ctx.bpe_ranks.empty());
  CHECK(ctx.ugm_ready);
}

TEST_CASE("encoder_assign_bpe_regex_variants") {
  vocab_builder builder{};
  builder.set_model("gpt2");

  emel::encoder::bpe::action::context ctx{};
  ctx.vocab = builder.vocab;

  builder.set_pre("gpt2");
  emel::encoder::bpe::detail::assign_bpe_regex(ctx, *builder.vocab);
  CHECK(ctx.bpe_pre_id == emel::model::data::tokenizer_pre::GPT2);
  CHECK(!ctx.bpe_regex_exprs.empty());

  builder.set_pre("llama3");
  emel::encoder::bpe::detail::assign_bpe_regex(ctx, *builder.vocab);
  CHECK(ctx.bpe_pre_id == emel::model::data::tokenizer_pre::LLAMA3);

  builder.set_pre("mpt");
  emel::encoder::bpe::detail::assign_bpe_regex(ctx, *builder.vocab);
  CHECK(ctx.bpe_pre_id == emel::model::data::tokenizer_pre::MPT);
}

TEST_CASE("encoder_guard_validates_inputs") {
  vocab_builder builder{};
  std::array<int32_t, 2> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  emel::encoder::event::encode valid{
    .vocab = builder.vocab,
    .text = "ok",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  emel::encoder::event::encode invalid{
    .vocab = builder.vocab,
    .text = "bad",
    .token_ids = nullptr,
    .token_capacity = 0,
    .token_count_out = &token_count,
    .error_out = &err,
  };

  emel::encoder::event::encode missing_count{
    .vocab = builder.vocab,
    .text = "bad",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = nullptr,
    .error_out = &err,
  };

  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;

  CHECK(emel::encoder::guard::valid_encode{}(valid, ctx));
  CHECK(!emel::encoder::guard::valid_encode{}(invalid, ctx));
  CHECK(!emel::encoder::guard::valid_encode{}(missing_count, ctx));

  emel::encoder::action::context empty_ctx{};
  CHECK(emel::encoder::guard::valid_encode{}(valid, empty_ctx));
}

TEST_CASE("encoder_detail_misc_branches") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t raw_x = builder.add_token("x", 0.0f, 1);
  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx));

  CHECK(emel::encoder::detail::byte_to_token(
    ctx, *builder.vocab, static_cast<uint8_t>('x'),
    emel::model::data::tokenizer_model::SPM) == raw_x);

  CHECK(emel::encoder::detail::token_text(*builder.vocab, -1).empty());
  CHECK(!emel::encoder::detail::is_token_type(*builder.vocab, -1, 1));

  const auto empty_view = emel::encoder::detail::string_view_from_array(
    std::array<char, 1>{{'\0'}});
  CHECK(empty_view.empty());
}

TEST_CASE("unicode_helpers_extra_branches") {
  const std::string invalid_utf8 = "\xC3\x28";
  const auto invalid_cpts = emel::text::unicode_cpts_from_utf8(invalid_utf8);
  CHECK(!invalid_cpts.empty());

  const std::string b = emel::text::unicode_byte_to_utf8(0x62);
  CHECK(emel::text::unicode_utf8_to_byte(b) == 0x62);

  const auto flags_empty = emel::text::unicode_cpt_flags_from_utf8(std::string{});
  CHECK(!flags_empty.is_letter);
  CHECK(!flags_empty.is_number);
  CHECK(!flags_empty.is_whitespace);
}

TEST_CASE("encoder_detail_helpers") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t hello_id = builder.add_token("hello", 0.5f, 1);
  const int32_t world_id = builder.add_token("world", 0.4f, 1);
  builder.add_byte_token(static_cast<uint8_t>('!'));

  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx));

  const auto view = emel::encoder::detail::string_view_from_array(
    std::array<char, 6>{{'h', 'e', 'l', 'l', 'o', '\0'}});
  CHECK(view == "hello");
  const auto view_full = emel::encoder::detail::string_view_from_array(
    std::array<char, 3>{{'x', 'y', 'z'}});
  CHECK(view_full.size() == 3);
  CHECK(emel::encoder::detail::utf8_len(static_cast<char>(0x7F)) == 1);
  CHECK(emel::encoder::detail::utf8_len(static_cast<char>(0xC2)) == 2);
  CHECK(emel::encoder::detail::utf8_len(static_cast<char>(0xE2)) == 3);
  CHECK(emel::encoder::detail::utf8_len(static_cast<char>(0xF0)) == 4);

  CHECK(emel::encoder::detail::token_text(*builder.vocab, hello_id) == "hello");
  CHECK(emel::encoder::detail::is_token_type(*builder.vocab, world_id, 1));

  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::BPE);
  CHECK(emel::encoder::detail::lookup_token(ctx, "hello") == hello_id);

  std::array<int32_t, 1> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "hello",
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  int32_t count = 0;
  CHECK(emel::encoder::detail::push_token(ev, hello_id, count));
  CHECK(count == 1);
  CHECK(!emel::encoder::detail::push_token(ev, world_id, count));

  std::vector<std::string_view> parts;
  emel::encoder::detail::split_whitespace("a  b\tc", parts);
  std::vector<std::string_view> non_empty;
  for (const auto part : parts) {
    if (!part.empty()) {
      non_empty.push_back(part);
    }
  }
  REQUIRE(non_empty.size() >= 3);
  auto has_token = [] (const std::vector<std::string_view> & values,
                        const std::string_view value) {
    for (const auto item : values) {
      if (item == value) {
        return true;
      }
    }
    return false;
  };
  CHECK(has_token(non_empty, "a"));
  CHECK(has_token(non_empty, "b"));
  CHECK(has_token(non_empty, "c"));

  const auto wpm_tokens = emel::encoder::wpm::detail::wpm_preprocess("hello!");
  CHECK(!wpm_tokens.empty());

  CHECK(emel::encoder::detail::is_chinese_char(0x4E00));
  CHECK(!emel::encoder::detail::is_chinese_char(0x0041));
  CHECK(!emel::encoder::detail::cpt_to_utf8(0x24).empty());
  CHECK(emel::encoder::detail::cpt_to_utf8(0x00A2).size() == 2);
  CHECK(emel::encoder::detail::cpt_to_utf8(0x20AC).size() == 3);
  CHECK(emel::encoder::detail::cpt_to_utf8(0x1F4A9).size() == 4);
  CHECK(emel::encoder::detail::byte_to_utf8_table()[static_cast<size_t>('A')] == "A");
  CHECK(!emel::encoder::detail::byte_to_utf8_table()[0x01].empty());

  emel::encoder::ugm::action::context ugm_ctx{};
  ugm_ctx.vocab = builder.vocab;
  std::string_view normalized{};
  CHECK(emel::encoder::ugm::detail::normalize_ugm_into(
    *builder.vocab, ugm_ctx, "hello", normalized));
  CHECK(!normalized.empty());
}

TEST_CASE("encoder_encode_impl_variants") {
  auto run_variant = [](
    const char * model,
    const char * pre,
    std::string_view text,
    const auto & setup_vocab) {
      vocab_builder builder{};
      builder.set_model(model);
      if (pre != nullptr) {
        builder.set_pre(pre);
      }
      setup_vocab(builder);

      std::array<int32_t, 32> out_tokens = {};
      int32_t token_count = 0;
      int32_t err = EMEL_OK;
      emel::encoder::event::encode ev{
        .text = text,
        .token_ids = out_tokens.data(),
        .token_capacity = static_cast<int32_t>(out_tokens.size()),
        .token_count_out = &token_count,
        .error_out = &err,
      };

      const auto model_id = builder.vocab->tokenizer_model_id;
      emel::encoder::detail::encode_result result{};
      switch (model_id) {
        case emel::model::data::tokenizer_model::SPM: {
          emel::encoder::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::encoder::detail::ensure_tables(ctx));
          result = emel::encoder::spm::detail::encode_spm(ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::BPE: {
          emel::encoder::bpe::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::encoder::detail::ensure_tables(ctx));
          if (pre != nullptr) {
            emel::encoder::bpe::detail::assign_bpe_regex(ctx, *builder.vocab);
          }
          result = emel::encoder::bpe::detail::encode_bpe(ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::WPM: {
          emel::encoder::wpm::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::encoder::detail::ensure_tables(ctx));
          result = emel::encoder::wpm::detail::encode_wpm(ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::UGM: {
          emel::encoder::ugm::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::encoder::detail::ensure_tables(ctx));
          result = emel::encoder::ugm::detail::encode_ugm(ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::RWKV: {
          emel::encoder::rwkv::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::encoder::detail::ensure_tables(ctx));
          result = emel::encoder::rwkv::detail::encode_rwkv(ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::PLAMO2: {
          emel::encoder::plamo2::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::encoder::detail::ensure_tables(ctx));
          result = emel::encoder::plamo2::detail::encode_plamo2(ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::UNKNOWN: {
          emel::encoder::action::context ctx{};
          ctx.vocab = builder.vocab;
          CHECK(emel::encoder::detail::ensure_tables(ctx));
          result = emel::encoder::fallback::detail::encode_fallback(ev, ctx, *builder.vocab);
          break;
        }
        case emel::model::data::tokenizer_model::NONE:
          result.error = EMEL_ERR_BACKEND;
          break;
      }
      if (ev.token_count_out != nullptr) {
        *ev.token_count_out = result.token_count;
      }
      if (ev.error_out != nullptr) {
        *ev.error_out = result.error;
      }
      (void)result;
      CHECK(err == EMEL_OK);
    };

  run_variant("gpt2", "gpt2", "hello world", [] (vocab_builder & builder) {
    builder.add_token("hello", 0.5f, 1);
    builder.add_token("world", 0.4f, 1);
    builder.add_token("h", 0.1f, 1);
    builder.add_token("e", 0.1f, 1);
    builder.add_merge("h e");
    builder.add_byte_token(static_cast<uint8_t>(' '));
  });

  run_variant("bert", nullptr, "hello", [] (vocab_builder & builder) {
    builder.add_token("he", 0.2f, 1);
    builder.add_token("##llo", 0.2f, 1);
    builder.add_token("<unk>", 0.0f, 2);
  });

  run_variant("t5", "gpt2", "hello world", [] (vocab_builder & builder) {
    builder.add_token("\xE2\x96\x81hello", 0.5f, 1);
    builder.add_token("world", 0.4f, 1);
    builder.set_charsmap_a_to_b();
  });

  run_variant("rwkv", nullptr, "r", [] (vocab_builder & builder) {
    builder.add_byte_token(static_cast<uint8_t>('r'));
  });

  run_variant("plamo2", nullptr, "p", [] (vocab_builder & builder) {
    builder.add_token("<unk>", 0.0f, 2);
    builder.add_all_plamo2_byte_tokens();
    builder.add_plamo2_byte_token(static_cast<uint8_t>('p'));
  });

  run_variant("unknown", nullptr, "x", [] (vocab_builder & builder) {
    builder.add_byte_token(static_cast<uint8_t>('x'));
  });
}

TEST_CASE("encoder_detail_encode_direct_calls") {
  std::array<int32_t, 32> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "hello world",
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  {
    vocab_builder builder{};
    builder.set_model("gpt2");
    builder.set_pre("gpt2");
    builder.add_token("h", 0.1f, 1);
    builder.add_token("e", 0.1f, 1);
    builder.add_merge("h e");
    builder.add_byte_token(static_cast<uint8_t>(' '));
    emel::encoder::bpe::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    emel::encoder::bpe::detail::assign_bpe_regex(ctx, *builder.vocab);
    emel::encoder::event::encode ev_plain = ev;
    auto result = emel::encoder::bpe::detail::encode_bpe(ev_plain, ctx, *builder.vocab);
    (void)result;
    emel::encoder::event::encode ev_punct = ev;
    ev_punct.text = "hello, world!";
    auto result_punct = emel::encoder::bpe::detail::encode_bpe(ev_punct, ctx, *builder.vocab);
    (void)result_punct;
    emel::encoder::event::encode ev_empty = ev;
    ev_empty.text = "";
    auto result_empty = emel::encoder::bpe::detail::encode_bpe(ev_empty, ctx, *builder.vocab);
    (void)result_empty;
  }

  {
    vocab_builder builder{};
    builder.set_model("bert");
    builder.add_token("he", 0.2f, 1);
    builder.add_token("##llo", 0.2f, 1);
    builder.add_token("<unk>", 0.0f, 2);
    emel::encoder::wpm::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    emel::encoder::event::encode ev_wpm = ev;
    ev_wpm.text = "unaffable";
    auto result = emel::encoder::wpm::detail::encode_wpm(ev_wpm, ctx, *builder.vocab);
    (void)result;
    emel::encoder::event::encode ev_unknown = ev;
    ev_unknown.text = "xyzxyz";
    auto result_unknown = emel::encoder::wpm::detail::encode_wpm(ev_unknown, ctx, *builder.vocab);
    (void)result_unknown;
  }

  {
    vocab_builder builder{};
    builder.set_model("t5");
    builder.set_pre("gpt2");
    builder.add_token("\xE2\x96\x81hello", 0.5f, 1);
    builder.add_token("world", 0.4f, 1);
    builder.set_charsmap_a_to_b();
    emel::encoder::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    emel::encoder::event::encode ev_spm = ev;
    ev_spm.text = "hello world";
    auto result = emel::encoder::spm::detail::encode_spm(ev_spm, ctx, *builder.vocab);
    (void)result;
  }

  {
    vocab_builder builder{};
    builder.set_model("t5");
    builder.add_token("\xE2\x96\x81hello", 0.5f, 1);
    builder.add_token("world", 0.4f, 1);
    emel::encoder::ugm::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    emel::encoder::event::encode ev_ugm = ev;
    ev_ugm.text = "hello";
    auto result = emel::encoder::ugm::detail::encode_ugm(ev_ugm, ctx, *builder.vocab);
    (void)result;
  }

  {
    vocab_builder builder{};
    builder.set_model("rwkv");
    builder.add_byte_token(static_cast<uint8_t>('r'));
    emel::encoder::rwkv::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    auto result = emel::encoder::rwkv::detail::encode_rwkv(ev, ctx, *builder.vocab);
    (void)result;
  }

  {
    vocab_builder builder{};
    builder.set_model("plamo2");
    builder.add_token("<unk>", 0.0f, 2);
    builder.add_all_plamo2_byte_tokens();
    builder.add_plamo2_byte_token(static_cast<uint8_t>('p'));
    emel::encoder::plamo2::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    auto result = emel::encoder::plamo2::detail::encode_plamo2(ev, ctx, *builder.vocab);
    (void)result;
  }

  {
    vocab_builder builder{};
    builder.set_model("unknown");
    builder.add_byte_token(static_cast<uint8_t>('x'));
    emel::encoder::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    auto result = emel::encoder::fallback::detail::encode_fallback(ev, ctx, *builder.vocab);
    (void)result;
  }
}

TEST_CASE("encoder_detail_branch_coverage") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t hex_id = builder.add_token("<0x41>", 0.0f, 1);
  const int32_t raw_id = builder.add_token("A", 0.0f, 1);
  const int32_t byte_id = builder.add_byte_token(static_cast<uint8_t>('!'));

  emel::encoder::bpe::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx));

  CHECK(emel::encoder::detail::byte_to_token(
    ctx, *builder.vocab, static_cast<uint8_t>('A'),
    emel::model::data::tokenizer_model::SPM) == hex_id);
  CHECK(emel::encoder::detail::byte_to_token(
    ctx, *builder.vocab, static_cast<uint8_t>('!'),
    emel::model::data::tokenizer_model::BPE) == byte_id);
  CHECK(emel::encoder::detail::byte_to_token(
    ctx, *builder.vocab, static_cast<uint8_t>('A'),
    emel::model::data::tokenizer_model::UNKNOWN) == raw_id);

  builder.set_model("t5");
  builder.vocab->escape_whitespaces = true;
  builder.vocab->treat_whitespace_as_suffix = false;
  builder.vocab->add_space_prefix = true;
  builder.vocab->remove_extra_whitespaces = true;
  emel::encoder::ugm::action::context ugm_ctx{};
  ugm_ctx.vocab = builder.vocab;
  std::string_view normalized{};
  CHECK(emel::encoder::ugm::detail::normalize_ugm_into(
    *builder.vocab, ugm_ctx, "  a", normalized));
  CHECK(!normalized.empty());

  builder.set_model("none");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::NONE);
  builder.set_model("no_vocab");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::NONE);
  builder.set_model("llama");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::SPM);
  builder.set_model("bert");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::WPM);
  builder.set_model("t5");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::UGM);
  builder.set_model("rwkv");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::RWKV);
  builder.set_model("plamo2");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::PLAMO2);
  builder.set_model("unknown");
  CHECK(builder.vocab->tokenizer_model_id == emel::model::data::tokenizer_model::UNKNOWN);

  builder.set_pre("");
  emel::encoder::bpe::detail::assign_bpe_regex(ctx, *builder.vocab);
}

TEST_CASE("encoder_detail_merge_and_token_helpers") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t token_id = builder.add_token("token", 0.2f, 1);
  const int32_t empty_id = builder.add_token("x", 0.1f, 1);
  builder.add_merge("a b");

  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx));

  CHECK(emel::encoder::detail::token_text(*builder.vocab, -1).empty());
  builder.vocab->entries[static_cast<uint32_t>(empty_id)].text_length = 0;
  CHECK(emel::encoder::detail::token_text(*builder.vocab, empty_id).empty());

  CHECK(emel::encoder::detail::merge_text(*builder.vocab, -1).empty());
  const uint32_t original_len = builder.vocab->merge_lengths[0];
  builder.vocab->merge_lengths[0] =
      static_cast<uint32_t>(builder.vocab->merge_storage.size() + 1);
  CHECK(emel::encoder::detail::merge_text(*builder.vocab, 0).empty());
  builder.vocab->merge_lengths[0] = original_len;
  CHECK(!emel::encoder::detail::merge_text(*builder.vocab, 0).empty());

  CHECK(!emel::encoder::detail::merge_match("", "a", "b"));
  CHECK(!emel::encoder::detail::merge_match("ab", "a", "b"));
  CHECK(!emel::encoder::detail::merge_match("b a", "a", "b"));
  CHECK(!emel::encoder::detail::merge_match("a b c", "a", "b"));
  CHECK(!emel::encoder::detail::merge_match("a x", "a", "b"));
  CHECK(emel::encoder::detail::merge_match("a b", "a", "b"));

  CHECK(emel::encoder::detail::hash_sv("token") != 0u);
  CHECK(emel::encoder::detail::hash_pair("a", "b") != 0u);

  emel::encoder::detail::token_map token_map{};
  CHECK(emel::encoder::detail::insert_token_map(
    token_map, *builder.vocab, "", token_id));
  CHECK(emel::encoder::detail::insert_token_map(
    token_map, *builder.vocab, "token", token_id));
  CHECK(emel::encoder::detail::insert_token_map(
    token_map, *builder.vocab, "token", token_id + 1));
  CHECK(token_map.count >= 1);

  emel::encoder::detail::merge_map merge_map{};
  CHECK(!emel::encoder::detail::insert_merge_map(
    merge_map, "", "b", 0, *builder.vocab));
  CHECK(emel::encoder::detail::insert_merge_map(
    merge_map, "a", "b", 0, *builder.vocab));
  CHECK(emel::encoder::detail::insert_merge_map(
    merge_map, "a", "b", 1, *builder.vocab));
  CHECK(merge_map.count >= 1);
}

TEST_CASE("encoder_detail_lookup_helpers") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  const int32_t token_a = builder.add_token("a", 0.1f, 1);
  const int32_t token_ab = builder.add_token("ab", 0.1f, 1);
  const int32_t token_ac = builder.add_token("ac", 0.1f, 1);
  const int32_t token_cb = builder.add_token("cb", 0.1f, 1);
  builder.add_merge("a b");

  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx));

  CHECK(emel::encoder::detail::lookup_token(ctx, "") ==
        emel::encoder::detail::k_token_null);
  CHECK(emel::encoder::detail::lookup_token(ctx, "missing") ==
        emel::encoder::detail::k_token_null);

  const uint32_t hash = emel::encoder::detail::hash_concat("a", "b");
  const uint32_t mask = emel::encoder::detail::k_token_hash_size - 1;
  const uint32_t slot = hash & mask;

  ctx.token_to_id.hashes[slot] = hash;
  ctx.token_to_id.values[slot] = token_cb;
  int32_t concat = emel::encoder::detail::lookup_token_concat(ctx, "a", "b");
  CHECK((concat == token_ab || concat == emel::encoder::detail::k_token_null));

  ctx.token_to_id.hashes[slot] = hash;
  ctx.token_to_id.values[slot] = token_ac;
  concat = emel::encoder::detail::lookup_token_concat(ctx, "a", "b");
  CHECK((concat == token_ab || concat == emel::encoder::detail::k_token_null));

  ctx.token_to_id.hashes[slot] = hash;
  ctx.token_to_id.values[slot] = token_a;
  concat = emel::encoder::detail::lookup_token_concat(ctx, "a", "b");
  CHECK((concat == token_ab || concat == emel::encoder::detail::k_token_null));

  CHECK(emel::encoder::detail::lookup_merge_rank(ctx, *builder.vocab, "", "b") ==
        emel::encoder::detail::k_token_null);
  CHECK(emel::encoder::detail::lookup_merge_rank(ctx, *builder.vocab, "a", "") ==
        emel::encoder::detail::k_token_null);
}

TEST_CASE("encoder_detail_ensure_tables_null_vocab") {
  emel::encoder::action::context ctx{};
  CHECK(!emel::encoder::detail::ensure_tables(ctx));
}

TEST_CASE("encoder_detail_empty_encode_variants") {
  vocab_builder builder{};
  builder.set_model("unknown");

  emel::encoder::action::context fallback_ctx{};
  fallback_ctx.vocab = builder.vocab;
  emel::encoder::rwkv::action::context rwkv_ctx{};
  rwkv_ctx.vocab = builder.vocab;
  emel::encoder::plamo2::action::context plamo2_ctx{};
  plamo2_ctx.vocab = builder.vocab;

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto fallback =
    emel::encoder::fallback::detail::encode_fallback(ev, fallback_ctx, *builder.vocab);
  CHECK(fallback.token_count == 0);
  CHECK(fallback.error == EMEL_OK);

  const auto rwkv = emel::encoder::rwkv::detail::encode_rwkv(ev, rwkv_ctx, *builder.vocab);
  CHECK(rwkv.token_count == 0);
  CHECK(rwkv.error == EMEL_OK);

  const auto plamo2 =
    emel::encoder::plamo2::detail::encode_plamo2(ev, plamo2_ctx, *builder.vocab);
  CHECK(plamo2.token_count == 0);
  CHECK(plamo2.error == EMEL_OK);
}

TEST_CASE("encoder_detail_encode_cpt_utf8_branches") {
  char out[4] = {};
  CHECK(emel::encoder::detail::encode_cpt_utf8(0x20AC, out) == 3);
  CHECK(emel::encoder::detail::encode_cpt_utf8(0x1F4A9, out) == 4);
}

TEST_CASE("encoder_detail_wpm_preprocess_whitespace") {
  const auto parts = emel::encoder::wpm::detail::wpm_preprocess("a b");
  CHECK(parts.size() >= 2);
  CHECK(parts[0] == "a");
  CHECK(parts[1] == "b");
}

TEST_CASE("encoder_detail_byte_to_token_none") {
  vocab_builder builder{};
  builder.set_model("none");
  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::byte_to_token(
    ctx, *builder.vocab, static_cast<uint8_t>('A'),
    emel::model::data::tokenizer_model::NONE) ==
    emel::encoder::detail::k_token_null);
}

TEST_CASE("encoder_detail_ensure_tables_merge_variants") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("a", 0.1f, 1);
  builder.add_merge("");
  builder.add_merge("nospace");

  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx));
}

TEST_CASE("encoder_detail_bpe_merge_and_errors") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("h", 0.1f, 1);
  builder.add_token("e", 0.1f, 1);
  const int32_t he_id = builder.add_token("he", 0.5f, 1);
  builder.add_merge("h e");

  emel::encoder::bpe::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx));

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "he",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto merged = emel::encoder::bpe::detail::encode_bpe(ev, ctx, *builder.vocab);
  CHECK(merged.error == EMEL_OK);
  CHECK(merged.token_count >= 1);
  CHECK(tokens[0] == he_id);

  builder.vocab->ignore_merges = true;
  emel::encoder::bpe::action::context ctx_fail{};
  ctx_fail.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx_fail));

  emel::encoder::event::encode ev_fail{
    .text = "he",
    .token_ids = nullptr,
    .token_capacity = 0,
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result_fail = emel::encoder::bpe::detail::encode_bpe(
    ev_fail, ctx_fail, *builder.vocab);
  CHECK(result_fail.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_xcda_error_paths") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.set_charsmap_a_to_b();

  emel::encoder::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;

  const uint32_t original_size = builder.vocab->precompiled_charsmap_size;
  builder.vocab->precompiled_charsmap_size = 2;
  CHECK(!emel::encoder::ugm::detail::init_xcda_tables(ctx));
  builder.vocab->precompiled_charsmap_size = original_size;
  CHECK(emel::encoder::ugm::detail::init_xcda_tables(ctx));
}

TEST_CASE("encoder_detail_spm_merge_capacity_error") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.set_pre("gpt2");
  builder.add_token("h", 0.1f, 1);
  builder.add_token("i", 0.1f, 1);
  builder.add_token("hi", 0.9f, 1);

  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx));

  std::array<int32_t, 1> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "hi",
    .token_ids = tokens.data(),
    .token_capacity = 0,
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_spm_add_space_prefix") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.set_pre("gpt2");
  builder.add_token("\xE2\x96\x81", 0.1f, 1);
  builder.add_token("\xE2\x96\x81h", 0.2f, 1);
  builder.add_token("\xE2\x96\x81hi", 0.9f, 1);
  builder.add_token("h", 0.1f, 1);
  builder.add_token("i", 0.1f, 1);
  builder.add_token(" ", 0.1f, 1);
  builder.vocab->add_space_prefix = true;

  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx));

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "hi",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count >= 1);
}

TEST_CASE("encoder_detail_bpe_buffer_overflow") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("a", 0.1f, 1);

  emel::encoder::bpe::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::detail::ensure_tables(ctx));

  std::string text(70000, 'a');
  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = text,
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::bpe::detail::encode_bpe(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_wpm_empty_text") {
  vocab_builder builder{};
  builder.set_model("bert");
  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "",
    .token_ids = tokens.data(),
    .token_capacity = static_cast<int32_t>(tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::wpm::detail::encode_wpm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count == 0);
}

TEST_CASE("encoder_detail_charsmap_into_paths") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.set_charsmap_a_to_b();

  emel::encoder::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::ugm::detail::ensure_ugm_tables(ctx, *builder.vocab));

  const std::string a_input = "a";
  const auto mapped_a =
    emel::encoder::ugm::detail::normalize_prefix(*builder.vocab, ctx, a_input, 0);
  REQUIRE(mapped_a.normalized != nullptr);
  CHECK(std::string_view(mapped_a.normalized, mapped_a.normalized_len) == "b");
  CHECK(mapped_a.consumed_input == 1);

  const std::string b_input = "b";
  const auto mapped_b =
    emel::encoder::ugm::detail::normalize_prefix(*builder.vocab, ctx, b_input, 0);
  REQUIRE(mapped_b.normalized != nullptr);
  CHECK(std::string_view(mapped_b.normalized, mapped_b.normalized_len) == "b");
  CHECK(mapped_b.consumed_input == 1);

  builder.vocab->precompiled_charsmap_size = 0;
  emel::encoder::ugm::action::context ctx_no_table{};
  ctx_no_table.vocab = builder.vocab;
  CHECK(!emel::encoder::ugm::detail::init_xcda_tables(ctx_no_table));
}

TEST_CASE("encoder_detail_normalize_ugm_into_paths") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.set_charsmap_a_to_b();
  builder.vocab->escape_whitespaces = true;
  builder.vocab->treat_whitespace_as_suffix = false;
  builder.vocab->add_space_prefix = true;
  builder.vocab->remove_extra_whitespaces = true;

  emel::encoder::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;

  std::string_view out;
  CHECK(emel::encoder::ugm::detail::normalize_ugm_into(
    *builder.vocab, ctx, "a   ", out));
  CHECK(!out.empty());
  CHECK(out.front() == '\xE2');

  std::string_view trimmed;
  CHECK(emel::encoder::ugm::detail::normalize_ugm_into(
    *builder.vocab, ctx, "  a  ", trimmed));
  CHECK(!trimmed.empty());
}

TEST_CASE("encoder_assign_bpe_regex_variants_extended") {
  vocab_builder builder{};
  builder.set_model("gpt2");

  const std::array<const char *, 42> presets = {{
    "llama3",
    "dbrx",
    "smaug",
    "deepseek-llm",
    "deepseek3-llm",
    "hunyuan-dense",
    "youtu",
    "deepseek-coder",
    "falcon",
    "starcoder",
    "refact",
    "command-r",
    "smollm",
    "codeshell",
    "exaone",
    "minerva",
    "gpt2",
    "mpt",
    "olmo",
    "jais",
    "trillion",
    "granite-docling",
    "stablelm2",
    "qwen2",
    "hunyuan",
    "solar-open",
    "qwen35",
    "poro",
    "bloom",
    "gpt3-finnish",
    "chatglm4",
    "viking",
    "tekken",
    "chameleon",
    "gpt4o",
    "minimax-m2",
    "kimi-k2",
    "superbpe",
    "bailingmoe",
    "seed-coder",
    "grok-2",
    "afmoe",
  }};

  for (const auto pre : presets) {
    builder.set_pre(pre);
    emel::encoder::bpe::action::context ctx{};
    ctx.vocab = builder.vocab;
    emel::encoder::bpe::detail::assign_bpe_regex(ctx, *builder.vocab);
    CHECK(ctx.bpe_pre_id == builder.vocab->tokenizer_pre_id);
    CHECK(!ctx.bpe_regex_exprs.empty());
  }
}

TEST_CASE("encoder_assign_bpe_regex_enum_cases") {
  using tokenizer_pre = emel::model::data::tokenizer_pre;
  vocab_builder builder{};
  builder.set_model("gpt2");

  const std::array<tokenizer_pre, 47> presets = {{
    tokenizer_pre::DEFAULT,
    tokenizer_pre::LLAMA3,
    tokenizer_pre::JAIS2,
    tokenizer_pre::DBRX,
    tokenizer_pre::SMAUG,
    tokenizer_pre::DEEPSEEK_LLM,
    tokenizer_pre::DEEPSEEK3_LLM,
    tokenizer_pre::HUNYUAN_DENSE,
    tokenizer_pre::JOYAI_LLM,
    tokenizer_pre::YOUTU,
    tokenizer_pre::DEEPSEEK_CODER,
    tokenizer_pre::FALCON,
    tokenizer_pre::STARCODER,
    tokenizer_pre::REFACT,
    tokenizer_pre::COMMAND_R,
    tokenizer_pre::SMOLLM,
    tokenizer_pre::CODESHELL,
    tokenizer_pre::EXAONE,
    tokenizer_pre::MINERVA,
    tokenizer_pre::GPT2,
    tokenizer_pre::MPT,
    tokenizer_pre::OLMO,
    tokenizer_pre::JAIS,
    tokenizer_pre::TRILLION,
    tokenizer_pre::GRANITE_DOCLING,
    tokenizer_pre::QWEN35,
    tokenizer_pre::STABLELM2,
    tokenizer_pre::QWEN2,
    tokenizer_pre::HUNYUAN,
    tokenizer_pre::SOLAR_OPEN,
    tokenizer_pre::PORO,
    tokenizer_pre::BLOOM,
    tokenizer_pre::GPT3_FINNISH,
    tokenizer_pre::CHATGLM4,
    tokenizer_pre::VIKING,
    tokenizer_pre::TEKKEN,
    tokenizer_pre::CHAMELEON,
    tokenizer_pre::GPT4O,
    tokenizer_pre::MINIMAX_M2,
    tokenizer_pre::TINY_AYA,
    tokenizer_pre::KIMI_K2,
    tokenizer_pre::SUPERBPE,
    tokenizer_pre::BAILINGMOE,
    tokenizer_pre::SEED_CODER,
    tokenizer_pre::GROK_2,
    tokenizer_pre::AFMOE,
    tokenizer_pre::EXAONE_MOE,
  }};

  for (const auto pre : presets) {
    builder.vocab->tokenizer_pre_id = pre;
    emel::encoder::bpe::action::context ctx{};
    ctx.vocab = builder.vocab;
    emel::encoder::bpe::detail::assign_bpe_regex(ctx, *builder.vocab);
    CHECK(ctx.bpe_pre_id == pre);
    CHECK(!ctx.bpe_regex_exprs.empty());
  }
}

TEST_CASE("encoder_detail_rwkv_unescape_branches") {
  std::string out;
  CHECK(emel::encoder::rwkv::detail::unescape_rwkv_token("plain", out));
  CHECK(out == "plain");
  CHECK(emel::encoder::rwkv::detail::unescape_rwkv_token("\\n\\t\\r", out));
  CHECK(out == std::string("\n\t\r"));
  CHECK(emel::encoder::rwkv::detail::unescape_rwkv_token("\\\\", out));
  CHECK(out == "\\");
  CHECK(emel::encoder::rwkv::detail::unescape_rwkv_token("\\x41\\x42", out));
  CHECK(out == "AB");
  CHECK_FALSE(emel::encoder::rwkv::detail::unescape_rwkv_token("\\x1", out));
}

TEST_CASE("encoder_rwkv_tables_reject_incomplete_hex") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  builder.add_token("\\x1", 0.0f, 1);
  emel::encoder::rwkv::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK_FALSE(emel::encoder::rwkv::detail::ensure_rwkv_tables(ctx, *builder.vocab));
}

TEST_CASE("encoder_rwkv_skips_unknown_without_unk") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  builder.add_token("a", 0.0f, 1);
  builder.vocab->unk_id = emel::encoder::detail::k_token_null;

  emel::encoder::rwkv::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "b",
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::rwkv::detail::encode_rwkv(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count == 0);
}

TEST_CASE("encoder_detail_wpm_preprocess_punctuation_and_control") {
  const std::string input = std::string("hi,") + "\xEF\xBF\xBD" + "\xE4\xB8\xAD";
  const auto parts = emel::encoder::wpm::detail::wpm_preprocess(input);
  CHECK(parts.size() == 3);
  CHECK(parts[0] == "hi");
  CHECK(parts[1] == ",");
  CHECK(parts[2] == std::string("\xE4\xB8\xAD"));
}

TEST_CASE("encoder_detail_wpm_skips_unknown_without_unk") {
  vocab_builder builder{};
  builder.set_model("bert");
  builder.add_token("hello", 0.0f, 1);
  builder.vocab->unk_id = emel::encoder::detail::k_token_null;

  emel::encoder::wpm::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "unknown",
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::wpm::detail::encode_wpm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count == 0);
}

TEST_CASE("encoder_detail_spm_prefix_overflow") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.vocab->add_space_prefix = true;
  emel::encoder::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  const size_t max_bytes = ctx.scratch.buffer.size();
  std::string text(max_bytes, 'a');
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = text,
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::encoder::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_spm_space_overflow") {
  vocab_builder builder{};
  builder.set_model("llama");
  emel::encoder::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  const size_t max_bytes = ctx.scratch.buffer.size();
  std::string text(max_bytes - 1, 'a');
  text.back() = ' ';
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = text,
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::encoder::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_spm_missing_byte_token") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.add_token("a", 0.0f, 1);
  emel::encoder::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "b",
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::encoder::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_BACKEND);
}

TEST_CASE("encoder_detail_plamo2_bom_and_missing_bytes") {
  vocab_builder builder{};
  builder.set_model("plamo2");
  builder.add_token("dummy", 0.0f, 1);
  builder.add_token("", 0.0f, 1);
  builder.add_all_plamo2_byte_tokens();
  emel::encoder::plamo2::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::plamo2::detail::ensure_plamo2_tables(ctx, *builder.vocab));
  CHECK(emel::encoder::plamo2::detail::ensure_plamo2_tables(ctx, *builder.vocab));
  std::array<int32_t, 8> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "\xEF\xBB\xBF" "a",
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::encoder::plamo2::detail::encode_plamo2(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count > 0);

  emel::encoder::event::encode ev_bom_only = ev;
  ev_bom_only.text = "\xEF\xBB\xBF";
  const auto bom_only =
      emel::encoder::plamo2::detail::encode_plamo2(ev_bom_only, ctx, *builder.vocab);
  CHECK(bom_only.error == EMEL_OK);
  CHECK(bom_only.token_count == 0);

  emel::encoder::event::encode ev_long = ev;
  const size_t max_len = ctx.cpts.size();
  std::string long_text(max_len + 1, 'a');
  ev_long.text = long_text;
  const auto too_long =
      emel::encoder::plamo2::detail::encode_plamo2(ev_long, ctx, *builder.vocab);
  CHECK(too_long.error == EMEL_ERR_INVALID_ARGUMENT);

  vocab_builder incomplete_builder{};
  incomplete_builder.set_model("plamo2");
  incomplete_builder.add_token("dummy", 0.0f, 1);
  incomplete_builder.add_plamo2_byte_token(static_cast<uint8_t>('a'));
  emel::encoder::plamo2::action::context ctx_incomplete{};
  ctx_incomplete.vocab = incomplete_builder.vocab;
  emel::encoder::event::encode ev_incomplete = ev;
  ev_incomplete.text = "a";
  const auto invalid =
      emel::encoder::plamo2::detail::encode_plamo2(ev_incomplete, ctx_incomplete,
                                                   *incomplete_builder.vocab);
  CHECK(invalid.error == EMEL_ERR_MODEL_INVALID);
}

TEST_CASE("encoder_detail_ugm_helper_branches") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.add_token("", 0.0f, 1);
  builder.add_token("user", 0.0f, 4);
  builder.add_token("a", 0.1f, 1);

  emel::encoder::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::ugm::detail::ensure_ugm_tables(ctx, *builder.vocab));
  CHECK(emel::encoder::ugm::detail::ensure_ugm_tables(ctx, *builder.vocab));

  emel::encoder::detail::naive_trie trie{};
  trie.insert("a", 1, 1);
  CHECK(emel::encoder::ugm::detail::trie_longest_prefix(trie, "a", 1) == 1);
  CHECK(emel::encoder::ugm::detail::trie_longest_prefix(trie, "b", 1) == 0);
  CHECK(emel::encoder::ugm::detail::trie_longest_prefix(trie, "a", 0) == 0);

  emel::encoder::ugm::detail::xcda_view view{};
  CHECK(view.node(1) == 0);

  const std::string input = "user";
  const auto norm_end =
      emel::encoder::ugm::detail::normalize_prefix(*builder.vocab, ctx, input,
                                                   input.size());
  CHECK(norm_end.normalized_len == 0);

  const auto norm_user =
      emel::encoder::ugm::detail::normalize_prefix(*builder.vocab, ctx, input, 0);
  CHECK(norm_user.consumed_input == input.size());

  const std::string bad(1, static_cast<char>(0x80));
  const auto norm_bad =
      emel::encoder::ugm::detail::normalize_prefix(*builder.vocab, ctx, bad, 0);
  CHECK(norm_bad.consumed_input == 1);
}

TEST_CASE("encoder_detail_ugm_normalize_overflow") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.add_token("a", 0.0f, 1);
  builder.vocab->add_space_prefix = true;
  builder.vocab->treat_whitespace_as_suffix = false;
  builder.vocab->remove_extra_whitespaces = false;
  builder.vocab->escape_whitespaces = false;

  emel::encoder::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  const size_t max_bytes = ctx.scratch.buffer.size();
  std::string text(max_bytes, 'a');
  std::string_view normalized;
  CHECK_FALSE(emel::encoder::ugm::detail::normalize_ugm_into(*builder.vocab, ctx,
                                                             text, normalized));

  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = text,
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::encoder::ugm::detail::encode_ugm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_ugm_normalize_empty") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.add_token("a", 0.0f, 1);
  builder.vocab->add_space_prefix = false;
  builder.vocab->treat_whitespace_as_suffix = false;
  builder.vocab->remove_extra_whitespaces = true;

  emel::encoder::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "   ",
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::encoder::ugm::detail::encode_ugm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count == 0);
}

TEST_CASE("encoder_detail_bpe_byte_push_overflow") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.set_pre("gpt2");
  builder.add_token("a", 0.0f, 1);
  builder.add_token("b", 0.0f, 1);
  builder.add_merge("a b");

  emel::encoder::bpe::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 1> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "ab",
    .token_ids = out_tokens.data(),
    .token_capacity = 0,
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::bpe::detail::encode_bpe(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_wpm_prefix_overflow") {
  vocab_builder builder{};
  builder.set_model("bert");
  emel::encoder::wpm::action::context ctx{};
  ctx.vocab = builder.vocab;
  const size_t max_bytes = ctx.scratch.buffer.size();
  std::string text(max_bytes, 'a');
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = text,
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::wpm::detail::encode_wpm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_wpm_push_overflow") {
  vocab_builder builder{};
  builder.set_model("bert");
  builder.add_token("\xE2\x96\x81" "a", 0.0f, 1);
  emel::encoder::wpm::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 1> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "a",
    .token_ids = out_tokens.data(),
    .token_capacity = 0,
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::wpm::detail::encode_wpm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_spm_empty_text") {
  vocab_builder builder{};
  builder.set_model("llama");
  emel::encoder::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "",
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count == 0);
}

TEST_CASE("encoder_detail_spm_symbol_overflow") {
  vocab_builder builder{};
  builder.set_model("llama");
  emel::encoder::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  const size_t max_symbols = ctx.scratch.offsets.size();
  std::string text(max_symbols + 1, 'a');
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = text,
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::encoder::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_insert_token_map_full") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  const int32_t token_x = builder.add_token("x", 0.0f, 1);
  const int32_t token_y = builder.add_token("y", 0.0f, 1);

  emel::encoder::detail::token_map map{};
  const std::string_view target = "y";
  const uint32_t hash = emel::encoder::detail::hash_sv(target);
  for (uint32_t i = 0; i < emel::encoder::detail::k_token_hash_size; ++i) {
    map.hashes.get()[i] = hash;
    map.values.get()[i] = token_x;
  }

  const bool ok =
      emel::encoder::detail::insert_token_map(map, *builder.vocab, target, token_y);
  CHECK_FALSE(ok);
}

TEST_CASE("encoder_detail_insert_merge_map_full") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  builder.add_merge("a b");

  emel::encoder::detail::merge_map map{};
  const std::string_view left = "x";
  const std::string_view right = "y";
  const uint32_t hash = emel::encoder::detail::hash_pair(left, right);
  for (uint32_t i = 0; i < emel::encoder::detail::k_merge_hash_size; ++i) {
    map.hashes.get()[i] = hash;
    map.values.get()[i] = 0;
  }

  const bool ok =
      emel::encoder::detail::insert_merge_map(map, left, right, 1, *builder.vocab);
  CHECK_FALSE(ok);
}

TEST_CASE("encoder_detail_lookup_token_full_probe") {
  vocab_builder builder{};
  builder.set_model("gpt2");
  const int32_t token_x = builder.add_token("x", 0.0f, 1);
  builder.add_token("y", 0.0f, 1);

  emel::encoder::action::context ctx{};
  ctx.vocab = builder.vocab;
  const std::string_view target = "y";
  const uint32_t hash = emel::encoder::detail::hash_sv(target);
  for (uint32_t i = 0; i < emel::encoder::detail::k_token_hash_size; ++i) {
    ctx.token_to_id.hashes.get()[i] = hash;
    ctx.token_to_id.values.get()[i] = token_x;
  }

  const int32_t id = emel::encoder::detail::lookup_token(ctx, target);
  CHECK(id == emel::encoder::detail::k_token_null);
}

TEST_CASE("encoder_rwkv_table_cache_and_empty_token") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  builder.add_token("", 0.0f, 1);
  builder.add_token("a", 0.0f, 1);
  emel::encoder::rwkv::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::encoder::rwkv::detail::ensure_rwkv_tables(ctx, *builder.vocab));
  CHECK(emel::encoder::rwkv::detail::ensure_rwkv_tables(ctx, *builder.vocab));
}

TEST_CASE("encoder_rwkv_encode_reports_invalid_table") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  builder.add_token("\\x1", 0.0f, 1);
  emel::encoder::rwkv::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 2> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "a",
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::encoder::rwkv::detail::encode_rwkv(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_ugm_append_space_and_overflow") {
  vocab_builder builder{};
  builder.set_model("t5");
  builder.add_token("a", 0.0f, 1);
  builder.vocab->add_space_prefix = true;
  builder.vocab->treat_whitespace_as_suffix = true;
  builder.vocab->remove_extra_whitespaces = false;
  builder.vocab->escape_whitespaces = false;

  emel::encoder::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::string_view normalized;
  CHECK(emel::encoder::ugm::detail::normalize_ugm_into(*builder.vocab, ctx, "a",
                                                       normalized));
  CHECK(!normalized.empty());
  CHECK(normalized.back() == ' ');

  const size_t max_bytes = ctx.scratch.buffer.size();
  std::string spaces(max_bytes + 1, ' ');
  CHECK_FALSE(emel::encoder::ugm::detail::normalize_ugm_into(*builder.vocab, ctx,
                                                             spaces, normalized));
}

TEST_CASE("encoder_detail_ugm_xcda_break_and_trie_paths") {
  emel::encoder::detail::naive_trie trie{};
  trie.insert("a", 1, 1);
  trie.insert("ab", 2, 2);
  CHECK(emel::encoder::ugm::detail::trie_longest_prefix(trie, "ac", 2) == 1);

  vocab_builder builder{};
  builder.set_model("t5");
  builder.add_token("a", 0.0f, 1);
  emel::encoder::ugm::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<uint32_t, 1> table = {0};
  ctx.xcda_table = table.data();
  ctx.xcda_table_size = table.size();
  ctx.prefix_replacements = "";
  ctx.prefix_replacements_size = 0;
  const std::string input(1, '\0');
  const auto norm = emel::encoder::ugm::detail::normalize_prefix(*builder.vocab, ctx,
                                                                 input, 0);
  CHECK(norm.consumed_input == 1);
}

TEST_CASE("encoder_rwkv_push_unk_overflow") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  const int32_t unk_id = builder.add_token("<unk>", 0.0f, 1);
  builder.vocab->unk_id = unk_id;
  emel::encoder::rwkv::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 1> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "z",
    .token_ids = out_tokens.data(),
    .token_capacity = 0,
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::encoder::rwkv::detail::encode_rwkv(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_encode_branch_cases") {
  std::array<int32_t, 8> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::encoder::event::encode ev{
    .text = "hello",
    .token_ids = out_tokens.data(),
    .token_capacity = static_cast<int32_t>(out_tokens.size()),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  {
    vocab_builder builder{};
    builder.set_model("gpt2");
    builder.set_pre("gpt2");
    builder.add_token("hello", 0.5f, 1);
    builder.vocab->ignore_merges = true;
    emel::encoder::bpe::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    ctx.bpe_regex_exprs.clear();
    auto result = emel::encoder::bpe::detail::encode_bpe(ev, ctx, *builder.vocab);
    (void)result;
  }

  {
    vocab_builder builder{};
    builder.set_model("t5");
    const int32_t unk_id = builder.add_token("<unk>", 0.0f, 2);
    builder.vocab->unk_id = unk_id;
    emel::encoder::ugm::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    emel::encoder::event::encode ev_ugm = ev;
    ev_ugm.text = "xyz";
    auto result = emel::encoder::ugm::detail::encode_ugm(ev_ugm, ctx, *builder.vocab);
    (void)result;
  }

  {
    vocab_builder builder{};
    builder.set_model("rwkv");
    const int32_t unk_id = builder.add_token("<unk>", 0.0f, 2);
    builder.vocab->unk_id = unk_id;
    emel::encoder::rwkv::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    emel::encoder::event::encode ev_rwkv = ev;
    ev_rwkv.text = "x";
    auto result = emel::encoder::rwkv::detail::encode_rwkv(ev_rwkv, ctx, *builder.vocab);
    (void)result;
  }

  {
    vocab_builder builder{};
    builder.set_model("unknown");
    emel::encoder::action::context ctx{};
    ctx.vocab = builder.vocab;
    CHECK(emel::encoder::detail::ensure_tables(ctx));
    CHECK(emel::encoder::detail::byte_to_token(
      ctx, *builder.vocab, static_cast<uint8_t>('x'),
      emel::model::data::tokenizer_model::NONE) == emel::encoder::detail::k_token_null);
  }
}

TEST_CASE("encoder_naive_trie_branches") {
  emel::encoder::detail::naive_trie trie{};
  trie.insert("", 0, 7);
  trie.insert("ab", 2, 11);
  trie.insert("ab", 2, 12);

  const auto * node_a = trie.traverse('a');
  REQUIRE(node_a != nullptr);
  const auto * node_b = node_a->traverse('b');
  REQUIRE(node_b != nullptr);
  CHECK(node_b->has_value);
  CHECK(node_b->value == 12);

  CHECK(trie.traverse('z') == nullptr);
}

TEST_CASE("encoder_bigram_comparators") {
  using spm_bigram = emel::encoder::detail::spm_bigram;
  std::priority_queue<spm_bigram, spm_bigram::queue_storage, spm_bigram::comparator> spm_queue;
  spm_bigram spm_a;
  spm_a.score = 0.5f;
  spm_a.left = 1;
  spm_bigram spm_b;
  spm_b.score = 1.0f;
  spm_b.left = 0;
  spm_queue.push(spm_a);
  spm_queue.push(spm_b);
  CHECK(spm_queue.top().score == 1.0f);

  using bpe_bigram = emel::encoder::detail::bpe_bigram;
  std::priority_queue<bpe_bigram, std::vector<bpe_bigram>, bpe_bigram::comparator> bpe_queue;
  bpe_bigram bpe_a;
  bpe_a.rank = 5;
  bpe_a.left = 2;
  bpe_bigram bpe_b;
  bpe_b.rank = 1;
  bpe_b.left = 3;
  bpe_queue.push(bpe_a);
  bpe_queue.push(bpe_b);
  CHECK(bpe_queue.top().rank == 1);
}

TEST_CASE("encoder_action_guard_wrapper_coverage") {
  std::array<int32_t, 2> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  auto make_event = [&](const char * text, const int32_t capacity,
                        const emel::model::data::vocab * vocab) {
    return emel::encoder::event::encode{
      .vocab = vocab,
      .text = text,
      .token_ids = tokens.data(),
      .token_capacity = capacity,
      .token_count_out = &token_count,
      .error_out = &err,
    };
  };

  auto make_invalid_event = [&](const emel::model::data::vocab * vocab) {
    return emel::encoder::event::encode{
      .vocab = vocab,
      .text = "x",
      .token_ids = nullptr,
      .token_capacity = 0,
      .token_count_out = &token_count,
      .error_out = &err,
    };
  };

  vocab_builder base_builder{};
  base_builder.set_model("gpt2");
  base_builder.set_pre("gpt2");
  base_builder.add_token("x", 0.1f, 1);

  emel::encoder::action::context base_ctx{};
  base_ctx.vocab = base_builder.vocab;
  base_ctx.phase_error = EMEL_ERR_BACKEND;
  base_ctx.last_error = EMEL_OK;
  emel::encoder::action::ensure_last_error(base_ctx);
  CHECK(base_ctx.last_error == EMEL_ERR_BACKEND);
  base_ctx.phase_error = EMEL_OK;
  base_ctx.last_error = EMEL_OK;
  emel::encoder::action::ensure_last_error(base_ctx);
  CHECK(base_ctx.last_error == EMEL_ERR_BACKEND);

  auto base_ev = make_event("x", 1, base_builder.vocab);
  emel::encoder::action::on_unexpected(base_ev, base_ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  emel::encoder::action::on_unexpected(emel::encoder::events::encoding_done{nullptr, 0}, base_ctx);

  auto invalid_ev = make_invalid_event(base_builder.vocab);
  CHECK(emel::encoder::guard::valid_encode{}(base_ev, base_ctx));
  CHECK(emel::encoder::guard::invalid_encode{}(invalid_ev, base_ctx));
  base_ctx.phase_error = EMEL_OK;
  CHECK(emel::encoder::guard::phase_ok{}(base_ctx));
  base_ctx.phase_error = EMEL_ERR_BACKEND;
  CHECK(emel::encoder::guard::phase_failed{}(base_ctx));

  {
    vocab_builder builder{};
    builder.set_model("gpt2");
    builder.set_pre("gpt2");
    builder.add_token("x", 0.1f, 1);
    builder.add_byte_token(static_cast<uint8_t>('x'));

    emel::encoder::bpe::action::context ctx{};
    ctx.vocab = builder.vocab;
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);

    emel::encoder::bpe::action::reject_invalid_encode(ev_ok, ctx);
    emel::encoder::bpe::action::run_encode(ctx);
    emel::encoder::bpe::action::mark_done(ctx);
    ctx.phase_error = EMEL_ERR_BACKEND;
    emel::encoder::bpe::action::ensure_last_error(ctx);
    emel::encoder::bpe::action::on_unexpected(ev_ok, ctx);
    emel::encoder::bpe::action::begin_encode(ev_error, ctx);

    CHECK(emel::encoder::bpe::guard::valid_encode{}(ev_ok, ctx));
    CHECK(emel::encoder::bpe::guard::invalid_encode{}(ev_invalid, ctx));
    ctx.phase_error = EMEL_OK;
    CHECK(emel::encoder::bpe::guard::phase_ok{}(ctx));
    ctx.phase_error = EMEL_ERR_BACKEND;
    CHECK(emel::encoder::bpe::guard::phase_failed{}(ctx));
  }

  {
    vocab_builder builder{};
    builder.set_model("bert");
    const int32_t unk_id = builder.add_token("<unk>", 0.0f, 2);
    builder.vocab->unk_id = unk_id;
    builder.add_token("x", 0.1f, 1);

    emel::encoder::wpm::action::context ctx{};
    ctx.vocab = builder.vocab;
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);

    emel::encoder::wpm::action::reject_invalid_encode(ev_ok, ctx);
    emel::encoder::wpm::action::run_encode(ctx);
    emel::encoder::wpm::action::mark_done(ctx);
    ctx.phase_error = EMEL_ERR_BACKEND;
    emel::encoder::wpm::action::ensure_last_error(ctx);
    emel::encoder::wpm::action::on_unexpected(ev_ok, ctx);
    emel::encoder::wpm::action::begin_encode(ev_error, ctx);

    CHECK(emel::encoder::wpm::guard::valid_encode{}(ev_ok, ctx));
    CHECK(emel::encoder::wpm::guard::invalid_encode{}(ev_invalid, ctx));
    ctx.phase_error = EMEL_OK;
    CHECK(emel::encoder::wpm::guard::phase_ok{}(ctx));
    ctx.phase_error = EMEL_ERR_BACKEND;
    CHECK(emel::encoder::wpm::guard::phase_failed{}(ctx));
  }

  {
    vocab_builder builder{};
    builder.set_model("llama");
    builder.set_pre("gpt2");
    builder.add_token("x", 0.1f, 1);

    emel::encoder::spm::action::context ctx{};
    ctx.vocab = builder.vocab;
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);

    emel::encoder::spm::action::reject_invalid_encode(ev_ok, ctx);
    emel::encoder::spm::action::run_encode(ctx);
    emel::encoder::spm::action::mark_done(ctx);
    ctx.phase_error = EMEL_ERR_BACKEND;
    emel::encoder::spm::action::ensure_last_error(ctx);
    emel::encoder::spm::action::on_unexpected(ev_ok, ctx);
    emel::encoder::spm::action::begin_encode(ev_error, ctx);

    CHECK(emel::encoder::spm::guard::valid_encode{}(ev_ok, ctx));
    CHECK(emel::encoder::spm::guard::invalid_encode{}(ev_invalid, ctx));
    ctx.phase_error = EMEL_OK;
    CHECK(emel::encoder::spm::guard::phase_ok{}(ctx));
    ctx.phase_error = EMEL_ERR_BACKEND;
    CHECK(emel::encoder::spm::guard::phase_failed{}(ctx));
  }

  {
    vocab_builder builder{};
    builder.set_model("t5");
    const int32_t unk_id = builder.add_token("<unk>", 0.0f, 2);
    builder.vocab->unk_id = unk_id;
    builder.add_token("x", 0.1f, 1);

    emel::encoder::ugm::action::context ctx{};
    ctx.vocab = builder.vocab;
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);

    emel::encoder::ugm::action::reject_invalid_encode(ev_ok, ctx);
    emel::encoder::ugm::action::run_encode(ctx);
    emel::encoder::ugm::action::mark_done(ctx);
    ctx.phase_error = EMEL_ERR_BACKEND;
    emel::encoder::ugm::action::ensure_last_error(ctx);
    emel::encoder::ugm::action::on_unexpected(ev_ok, ctx);
    emel::encoder::ugm::action::begin_encode(ev_error, ctx);

    CHECK(emel::encoder::ugm::guard::valid_encode{}(ev_ok, ctx));
    CHECK(emel::encoder::ugm::guard::invalid_encode{}(ev_invalid, ctx));
    ctx.phase_error = EMEL_OK;
    CHECK(emel::encoder::ugm::guard::phase_ok{}(ctx));
    ctx.phase_error = EMEL_ERR_BACKEND;
    CHECK(emel::encoder::ugm::guard::phase_failed{}(ctx));
  }

  {
    vocab_builder builder{};
    builder.set_model("rwkv");
    builder.add_byte_token(static_cast<uint8_t>('x'));

    emel::encoder::rwkv::action::context ctx{};
    ctx.vocab = builder.vocab;
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);

    emel::encoder::rwkv::action::reject_invalid_encode(ev_ok, ctx);
    emel::encoder::rwkv::action::run_encode(ctx);
    emel::encoder::rwkv::action::mark_done(ctx);
    ctx.phase_error = EMEL_ERR_BACKEND;
    emel::encoder::rwkv::action::ensure_last_error(ctx);
    emel::encoder::rwkv::action::on_unexpected(ev_ok, ctx);
    emel::encoder::rwkv::action::begin_encode(ev_error, ctx);

    CHECK(emel::encoder::rwkv::guard::valid_encode{}(ev_ok, ctx));
    CHECK(emel::encoder::rwkv::guard::invalid_encode{}(ev_invalid, ctx));
    ctx.phase_error = EMEL_OK;
    CHECK(emel::encoder::rwkv::guard::phase_ok{}(ctx));
    ctx.phase_error = EMEL_ERR_BACKEND;
    CHECK(emel::encoder::rwkv::guard::phase_failed{}(ctx));
  }

  {
    vocab_builder builder{};
    builder.set_model("plamo2");
    builder.add_token("<unk>", 0.0f, 2);
    builder.add_all_plamo2_byte_tokens();
    builder.add_plamo2_byte_token(static_cast<uint8_t>('x'));

    emel::encoder::plamo2::action::context ctx{};
    ctx.vocab = builder.vocab;
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);

    emel::encoder::plamo2::action::reject_invalid_encode(ev_ok, ctx);
    emel::encoder::plamo2::action::run_encode(ctx);
    emel::encoder::plamo2::action::mark_done(ctx);
    ctx.phase_error = EMEL_ERR_BACKEND;
    emel::encoder::plamo2::action::ensure_last_error(ctx);
    emel::encoder::plamo2::action::on_unexpected(ev_ok, ctx);
    emel::encoder::plamo2::action::begin_encode(ev_error, ctx);

    CHECK(emel::encoder::plamo2::guard::valid_encode{}(ev_ok, ctx));
    CHECK(emel::encoder::plamo2::guard::invalid_encode{}(ev_invalid, ctx));
    ctx.phase_error = EMEL_OK;
    CHECK(emel::encoder::plamo2::guard::phase_ok{}(ctx));
    ctx.phase_error = EMEL_ERR_BACKEND;
    CHECK(emel::encoder::plamo2::guard::phase_failed{}(ctx));
  }

  {
    vocab_builder builder{};
    builder.set_model("unknown");
    builder.add_byte_token(static_cast<uint8_t>('x'));

    emel::encoder::fallback::action::context ctx{};
    ctx.vocab = builder.vocab;
    auto ev_ok = make_event("x", 1, builder.vocab);
    auto ev_error = make_event("x", 0, builder.vocab);
    auto ev_invalid = make_invalid_event(builder.vocab);

    emel::encoder::fallback::action::reject_invalid_encode(ev_ok, ctx);
    emel::encoder::fallback::action::run_encode(ctx);
    emel::encoder::fallback::action::mark_done(ctx);
    ctx.phase_error = EMEL_ERR_BACKEND;
    emel::encoder::fallback::action::ensure_last_error(ctx);
    emel::encoder::fallback::action::on_unexpected(ev_ok, ctx);
    emel::encoder::fallback::action::begin_encode(ev_error, ctx);

    CHECK(emel::encoder::fallback::guard::valid_encode{}(ev_ok, ctx));
    CHECK(emel::encoder::fallback::guard::invalid_encode{}(ev_invalid, ctx));
    ctx.phase_error = EMEL_OK;
    CHECK(emel::encoder::fallback::guard::phase_ok{}(ctx));
    ctx.phase_error = EMEL_ERR_BACKEND;
    CHECK(emel::encoder::fallback::guard::phase_failed{}(ctx));
  }
}
