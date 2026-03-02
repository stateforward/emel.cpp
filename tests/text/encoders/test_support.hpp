#pragma once

#include <array>
#include <queue>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>
#include <doctest/doctest.h>

#include "emel/text/encoders/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/bpe/detail.hpp"
#include "emel/text/encoders/spm/detail.hpp"
#include "emel/text/encoders/wpm/detail.hpp"
#include "emel/text/encoders/ugm/detail.hpp"
#include "emel/text/encoders/rwkv/detail.hpp"
#include "emel/text/encoders/plamo2/detail.hpp"
#include "emel/text/encoders/fallback/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/text/encoders/guards.hpp"
#include "emel/text/encoders/bpe/sm.hpp"
#include "emel/text/encoders/spm/sm.hpp"
#include "emel/text/encoders/wpm/sm.hpp"
#include "emel/text/encoders/ugm/sm.hpp"
#include "emel/text/encoders/rwkv/sm.hpp"
#include "emel/text/encoders/plamo2/sm.hpp"
#include "emel/text/encoders/fallback/sm.hpp"
#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"

namespace {

emel::model::data::vocab & vocab_storage() {
  static emel::model::data::vocab storage{};
  return storage;
}

[[maybe_unused]] size_t sum_offsets(const std::vector<size_t> & offsets) {
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

[[maybe_unused]] bool record_done(void * owner,
                                  const emel::text::encoders::events::encoding_done &) {
  if (owner == nullptr) {
    return false;
  }
  static_cast<dispatch_recorder *>(owner)->done_count += 1;
  return true;
}

[[maybe_unused]] bool record_error(void * owner,
                                   const emel::text::encoders::events::encoding_error &) {
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
