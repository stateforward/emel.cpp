#include <array>
#include <cstddef>
#include <cstring>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/tokenizer/sm.hpp"

namespace {

int32_t add_token(emel::model::data::vocab & vocab, const char * text,
                  int32_t type = 0) {
  const uint32_t len = static_cast<uint32_t>(std::strlen(text));
  const uint32_t offset = vocab.token_bytes_used;
  std::memcpy(vocab.token_storage.data() + offset, text, len);
  const uint32_t id = vocab.n_tokens;
  vocab.entries[id].text_offset = offset;
  vocab.entries[id].text_length = len;
  vocab.entries[id].score = 0.0f;
  vocab.entries[id].type = type;
  vocab.token_bytes_used += len;
  vocab.n_tokens = id + 1;
  return static_cast<int32_t>(id);
}

emel::model::data::vocab make_bpe_vocab() {
  emel::model::data::vocab vocab = {};
  vocab.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  vocab.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  vocab.ignore_merges = true;
  vocab.add_bos = true;
  vocab.add_eos = true;

  const int32_t hello_id = add_token(vocab, "hello");
  const int32_t world_id = add_token(vocab, "\xC4\xA0" "world");
  const int32_t bos_id = add_token(vocab, "<bos>");
  const int32_t eos_id = add_token(vocab, "<eos>");

  CHECK(hello_id == 0);
  CHECK(world_id == 1);
  vocab.bos_id = bos_id;
  vocab.eos_id = eos_id;
  return vocab;
}

}  // namespace

TEST_CASE("tokenizer_bind_and_tokenize_bpe") {
  emel::model::data::vocab vocab = make_bpe_vocab();
  emel::tokenizer::sm machine{};

  int32_t bind_err = EMEL_OK;
  emel::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.error_out = &bind_err;

  CHECK(machine.process_event(bind_ev));
  CHECK(bind_err == EMEL_OK);

  std::array<int32_t, 8> tokens = {};
  int32_t count = 0;
  int32_t tok_err = EMEL_OK;
  emel::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &vocab;
  tok_ev.text = std::string_view("hello world");
  tok_ev.add_special = true;
  tok_ev.parse_special = false;
  tok_ev.token_ids_out = tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(tokens.size());
  tok_ev.token_count_out = &count;
  tok_ev.error_out = &tok_err;

  CHECK(machine.process_event(tok_ev));
  CHECK(tok_err == EMEL_OK);
  CHECK(count == 4);
  CHECK(tokens[0] == vocab.bos_id);
  CHECK(tokens[1] == 0);
  CHECK(tokens[2] == 1);
  CHECK(tokens[3] == vocab.eos_id);
}

TEST_CASE("tokenizer_tokenize_requires_bind") {
  emel::model::data::vocab vocab = make_bpe_vocab();
  emel::tokenizer::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t count = 0;
  int32_t err = EMEL_OK;
  emel::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &vocab;
  tok_ev.text = std::string_view("hello");
  tok_ev.add_special = false;
  tok_ev.parse_special = false;
  tok_ev.token_ids_out = tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(tokens.size());
  tok_ev.token_count_out = &count;
  tok_ev.error_out = &err;

  CHECK_FALSE(machine.process_event(tok_ev));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(count == 0);
}

TEST_CASE("tokenizer_tokenize_rejects_mismatched_vocab") {
  emel::model::data::vocab vocab = make_bpe_vocab();
  emel::model::data::vocab other_vocab = make_bpe_vocab();
  emel::tokenizer::sm machine{};

  int32_t bind_err = EMEL_OK;
  emel::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.error_out = &bind_err;
  CHECK(machine.process_event(bind_ev));
  CHECK(bind_err == EMEL_OK);

  std::array<int32_t, 4> tokens = {};
  int32_t count = 0;
  int32_t err = EMEL_OK;
  emel::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &other_vocab;
  tok_ev.text = std::string_view("hello");
  tok_ev.add_special = false;
  tok_ev.parse_special = false;
  tok_ev.token_ids_out = tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(tokens.size());
  tok_ev.token_count_out = &count;
  tok_ev.error_out = &err;

  CHECK_FALSE(machine.process_event(tok_ev));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(count == 0);
}
