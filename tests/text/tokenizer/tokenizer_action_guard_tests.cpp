#include <array>
#include <cstring>

#include <doctest/doctest.h>

#include "emel/model/data.hpp"
#include "emel/text/tokenizer/actions.hpp"
#include "emel/text/tokenizer/guards.hpp"

namespace {

emel::model::data::vocab &make_vocab_for_specials() {
  static emel::model::data::vocab vocab = {};
  std::memset(&vocab, 0, sizeof(vocab));
  vocab.add_bos = true;
  vocab.add_eos = true;
  vocab.add_sep = true;
  vocab.bos_id = 1;
  vocab.eos_id = 2;
  vocab.sep_id = 3;
  return vocab;
}

} // namespace

TEST_CASE("tokenizer_guard_can_bind_requires_explicit_valid_variants") {
  auto &vocab = make_vocab_for_specials();
  emel::text::tokenizer::event::bind bind_ev = {};

  CHECK_FALSE(emel::text::tokenizer::guard::can_bind{}(bind_ev));

  bind_ev.vocab = &vocab;
  bind_ev.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
  bind_ev.encoder_variant = emel::text::encoders::encoder_kind::spm;
  CHECK(emel::text::tokenizer::guard::can_bind{}(bind_ev));

  emel::text::tokenizer::event::bind_ctx bind_ctx = {};
  emel::text::tokenizer::event::bind_runtime bind_runtime{bind_ev, bind_ctx};
  CHECK(emel::text::tokenizer::guard::can_bind{}(bind_runtime));

  bind_ev.preprocessor_variant =
      static_cast<emel::text::tokenizer::preprocessor::preprocessor_kind>(255);
  CHECK_FALSE(emel::text::tokenizer::guard::can_bind{}(bind_ev));

  bind_ev.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
  bind_ev.encoder_variant =
      static_cast<emel::text::encoders::encoder_kind>(255);
  CHECK_FALSE(emel::text::tokenizer::guard::can_bind{}(bind_ev));
}

TEST_CASE("tokenizer_guard_can_tokenize") {
  auto &vocab = make_vocab_for_specials();
  static emel::model::data::vocab other_vocab = {};
  std::memset(&other_vocab, 0, sizeof(other_vocab));

  emel::text::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;
  ctx.is_bound = true;

  std::array<int32_t, 2> tokens = {};
  int32_t count = 0;
  emel::text::tokenizer::event::tokenize ev = {};
  ev.vocab = &vocab;
  ev.token_ids_out = tokens.data();
  ev.token_capacity = static_cast<int32_t>(tokens.size());
  ev.token_count_out = &count;

  emel::text::tokenizer::event::tokenize_ctx runtime_ctx = {};
  emel::text::tokenizer::event::tokenize_runtime runtime_ev{ev, runtime_ctx};

  CHECK(emel::text::tokenizer::guard::can_tokenize{}(ev, ctx));
  CHECK(emel::text::tokenizer::guard::can_tokenize{}(runtime_ev, ctx));

  ev.vocab = nullptr;
  CHECK_FALSE(emel::text::tokenizer::guard::can_tokenize{}(ev, ctx));
  CHECK_FALSE(emel::text::tokenizer::guard::can_tokenize{}(runtime_ev, ctx));

  ev.vocab = &other_vocab;
  CHECK_FALSE(emel::text::tokenizer::guard::can_tokenize{}(runtime_ev, ctx));

  ev.vocab = &vocab;
  ev.token_ids_out = nullptr;
  CHECK_FALSE(emel::text::tokenizer::guard::can_tokenize{}(runtime_ev, ctx));

  ev.token_ids_out = tokens.data();
  ev.token_count_out = nullptr;
  CHECK_FALSE(emel::text::tokenizer::guard::can_tokenize{}(runtime_ev, ctx));
}

TEST_CASE("tokenizer_guard_prefix_suffix_cases") {
  auto &vocab = make_vocab_for_specials();
  emel::text::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;
  ctx.model_kind = emel::text::tokenizer::action::encoder_kind::bpe;

  std::array<int32_t, 1> tokens = {};
  int32_t token_count_out = 0;
  emel::text::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &vocab;
  tok_ev.add_special = true;
  tok_ev.token_ids_out = tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(tokens.size());
  tok_ev.token_count_out = &token_count_out;

  emel::text::tokenizer::event::tokenize_ctx tok_ctx = {};
  emel::text::tokenizer::event::tokenize_runtime runtime_ev{tok_ev, tok_ctx};

  CHECK(emel::text::tokenizer::guard::bos_ready{}(runtime_ev, ctx));
  CHECK_FALSE(emel::text::tokenizer::guard::bos_invalid_id{}(runtime_ev, ctx));
  CHECK_FALSE(emel::text::tokenizer::guard::bos_no_capacity{}(runtime_ev, ctx));

  tok_ctx.token_count = tok_ev.token_capacity;
  CHECK(emel::text::tokenizer::guard::bos_no_capacity{}(runtime_ev, ctx));

  tok_ctx.token_count = 0;
  vocab.bos_id = -1;
  CHECK(emel::text::tokenizer::guard::bos_invalid_id{}(runtime_ev, ctx));

  vocab.bos_id = 1;
  ctx.model_kind = emel::text::tokenizer::action::encoder_kind::wpm;
  CHECK(emel::text::tokenizer::guard::sep_ready{}(runtime_ev, ctx));
  tok_ctx.token_count = tok_ev.token_capacity;
  CHECK(emel::text::tokenizer::guard::sep_no_capacity{}(runtime_ev, ctx));

  tok_ctx.token_count = 0;
  vocab.sep_id = -1;
  CHECK(emel::text::tokenizer::guard::sep_invalid_id{}(runtime_ev, ctx));

  vocab.sep_id = 3;
  ctx.model_kind = emel::text::tokenizer::action::encoder_kind::bpe;
  CHECK(emel::text::tokenizer::guard::eos_ready{}(runtime_ev, ctx));
  tok_ctx.token_count = tok_ev.token_capacity;
  CHECK(emel::text::tokenizer::guard::eos_no_capacity{}(runtime_ev, ctx));

  tok_ctx.token_count = 0;
  vocab.eos_id = -1;
  CHECK(emel::text::tokenizer::guard::eos_invalid_id{}(runtime_ev, ctx));
}

TEST_CASE("tokenizer_actions_bind_paths") {
  auto &vocab = make_vocab_for_specials();
  emel::text::tokenizer::action::context ctx{};

  emel::text::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  bind_ev.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::bpe;
  bind_ev.encoder_variant = emel::text::encoders::encoder_kind::bpe;
  emel::text::tokenizer::event::bind_ctx bind_ctx = {};
  emel::text::tokenizer::event::bind_runtime bind_runtime{bind_ev, bind_ctx};

  emel::text::tokenizer::action::begin_bind(bind_runtime, ctx);
  CHECK(ctx.vocab == &vocab);
  CHECK_FALSE(ctx.is_bound);
  CHECK(bind_ctx.err ==
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none));

  emel::text::tokenizer::action::bind_preprocessor(bind_runtime, ctx);
  CHECK(bind_ctx.err ==
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none));

  emel::text::tokenizer::action::bind_encoder(bind_runtime, ctx);
  CHECK(bind_ctx.err ==
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none));

  emel::text::tokenizer::action::mark_bind_success(bind_runtime, ctx);
  CHECK(ctx.is_bound);
  CHECK(bind_ctx.result);

  emel::text::tokenizer::action::reject_bind(bind_runtime, ctx);
  CHECK(bind_ctx.err == emel::text::tokenizer::error_code(
                            emel::text::tokenizer::error::invalid_request));
  CHECK_FALSE(ctx.is_bound);
}

TEST_CASE("tokenizer_preprocess_decision_guards") {
  auto &vocab = make_vocab_for_specials();
  emel::text::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;

  std::array<int32_t, 2> tokens = {};
  int32_t token_count = 0;
  emel::text::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &vocab;
  tok_ev.text = "hello";
  tok_ev.token_ids_out = tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(tokens.size());
  tok_ev.token_count_out = &token_count;

  std::array<emel::text::tokenizer::action::fragment, 2> fragments = {};
  emel::text::tokenizer::event::tokenize_ctx tok_ctx = {};
  tok_ctx.fragments = fragments.data();
  tok_ctx.fragment_capacity = fragments.size();
  emel::text::tokenizer::event::tokenize_runtime runtime_ev{tok_ev, tok_ctx};

  tok_ctx.preprocess_accepted = false;
  tok_ctx.preprocess_err_code =
      emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  CHECK(
      emel::text::tokenizer::guard::preprocess_rejected_no_error{}(runtime_ev));

  tok_ctx.preprocess_accepted = true;
  tok_ctx.preprocess_err_code = emel::text::tokenizer::error_code(
      emel::text::tokenizer::error::model_invalid);
  CHECK(emel::text::tokenizer::guard::preprocess_reported_error{}(runtime_ev));

  tok_ctx.preprocess_accepted = true;
  tok_ctx.preprocess_err_code =
      emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  tok_ctx.fragment_count = tok_ctx.fragment_capacity + 1;
  CHECK(emel::text::tokenizer::guard::preprocess_fragment_count_invalid{}(
      runtime_ev));

  tok_ctx.fragment_count = 1;
  CHECK(emel::text::tokenizer::guard::preprocess_success{}(runtime_ev));
}

TEST_CASE("tokenizer_encode_decision_guards") {
  auto &vocab = make_vocab_for_specials();
  emel::text::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  emel::text::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &vocab;
  tok_ev.text = "hello";
  tok_ev.token_ids_out = tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(tokens.size());
  tok_ev.token_count_out = &token_count;

  emel::text::tokenizer::event::tokenize_ctx tok_ctx = {};
  emel::text::tokenizer::event::tokenize_runtime runtime_ev{tok_ev, tok_ctx};

  tok_ctx.encode_accepted = false;
  tok_ctx.encode_err_code =
      emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  CHECK(emel::text::tokenizer::guard::encode_rejected_no_error{}(runtime_ev));

  tok_ctx.encode_accepted = true;
  tok_ctx.encode_err_code = emel::text::tokenizer::error_code(
      emel::text::tokenizer::error::model_invalid);
  CHECK(emel::text::tokenizer::guard::encode_reported_error{}(runtime_ev));

  tok_ctx.encode_accepted = true;
  tok_ctx.encode_err_code =
      emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  tok_ctx.token_count = 3;
  tok_ctx.encode_token_count = 2;
  CHECK(emel::text::tokenizer::guard::encode_count_invalid{}(runtime_ev));

  tok_ctx.token_count = 1;
  tok_ctx.encode_token_count = 2;
  CHECK(emel::text::tokenizer::guard::encode_success{}(runtime_ev));
}

TEST_CASE("tokenizer_fragment_guards_and_append_action") {
  auto &vocab = make_vocab_for_specials();
  emel::text::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;

  std::array<int32_t, 4> tokens = {};
  int32_t count = 0;
  emel::text::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &vocab;
  tok_ev.token_ids_out = tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(tokens.size());
  tok_ev.token_count_out = &count;

  std::array<emel::text::tokenizer::action::fragment, 2> fragments = {};
  emel::text::tokenizer::event::tokenize_ctx tok_ctx = {};
  tok_ctx.fragments = fragments.data();
  tok_ctx.fragment_capacity = fragments.size();
  emel::text::tokenizer::event::tokenize_runtime runtime_ev{tok_ev, tok_ctx};

  tok_ctx.fragment_count = 1;
  tok_ctx.fragment_index = 0;
  tok_ctx.fragments[0].kind =
      emel::text::tokenizer::action::fragment_kind::token;
  tok_ctx.fragments[0].token = 7;

  CHECK(emel::text::tokenizer::guard::more_fragments_token_valid{}(runtime_ev));
  CHECK_FALSE(
      emel::text::tokenizer::guard::more_fragments_token_invalid{}(runtime_ev));
  CHECK_FALSE(emel::text::tokenizer::guard::more_fragments_raw{}(runtime_ev));

  emel::text::tokenizer::action::append_fragment_token(runtime_ev, ctx);
  CHECK(tok_ctx.token_count == 1);
  CHECK(tokens[0] == 7);
  CHECK(tok_ctx.fragment_index == 1);

  tok_ctx.fragment_count = 1;
  tok_ctx.fragment_index = 0;
  tok_ctx.token_count = 0;
  tok_ctx.fragments[0].kind =
      emel::text::tokenizer::action::fragment_kind::token;
  tok_ctx.fragments[0].token = -1;
  CHECK(
      emel::text::tokenizer::guard::more_fragments_token_invalid{}(runtime_ev));

  tok_ctx.fragments[0].kind =
      emel::text::tokenizer::action::fragment_kind::raw_text;
  CHECK(emel::text::tokenizer::guard::more_fragments_raw{}(runtime_ev));
}

TEST_CASE("tokenizer_actions_status_helpers") {
  auto &vocab = make_vocab_for_specials();
  emel::text::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;

  std::array<int32_t, 4> out_tokens = {};
  int32_t count = 0;
  emel::text::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = &vocab;
  tok_ev.token_ids_out = out_tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(out_tokens.size());
  tok_ev.token_count_out = &count;

  emel::text::tokenizer::event::tokenize_ctx tok_ctx = {};
  emel::text::tokenizer::event::tokenize_runtime tok_runtime{tok_ev, tok_ctx};

  emel::text::tokenizer::action::begin_tokenize(tok_runtime, ctx);
  CHECK(tok_ctx.token_count == 0);
  CHECK(tok_ctx.fragment_count == 0);

  emel::text::tokenizer::action::append_bos(tok_runtime, ctx);
  emel::text::tokenizer::action::append_sep(tok_runtime, ctx);
  emel::text::tokenizer::action::append_eos(tok_runtime, ctx);
  CHECK(tok_ctx.token_count == 3);
  CHECK(out_tokens[0] == vocab.bos_id);
  CHECK(out_tokens[1] == vocab.sep_id);
  CHECK(out_tokens[2] == vocab.eos_id);

  tok_ctx.preprocess_err_code = emel::text::tokenizer::error_code(
      emel::text::tokenizer::error::model_invalid);
  emel::text::tokenizer::action::set_error_from_preprocess(tok_runtime, ctx);
  CHECK(tok_ctx.err == emel::text::tokenizer::error_code(
                           emel::text::tokenizer::error::model_invalid));

  tok_ctx.encode_err_code = emel::text::tokenizer::error_code(
      emel::text::tokenizer::error::backend_error);
  emel::text::tokenizer::action::set_error_from_encode(tok_runtime, ctx);
  CHECK(tok_ctx.err == emel::text::tokenizer::error_code(
                           emel::text::tokenizer::error::backend_error));

  tok_ctx.encode_token_count = 1;
  tok_ctx.fragment_index = 0;
  emel::text::tokenizer::action::commit_encoded_fragment(tok_runtime, ctx);
  CHECK(tok_ctx.token_count == 4);
  CHECK(tok_ctx.fragment_index == 1);

  emel::text::tokenizer::action::set_backend_error(tok_runtime, ctx);
  CHECK(tok_ctx.err == emel::text::tokenizer::error_code(
                           emel::text::tokenizer::error::backend_error));

  emel::text::tokenizer::action::set_invalid_request_error(tok_runtime, ctx);
  CHECK(tok_ctx.err == emel::text::tokenizer::error_code(
                           emel::text::tokenizer::error::invalid_request));

  emel::text::tokenizer::action::set_invalid_id_error(tok_runtime, ctx);
  CHECK(tok_ctx.err == emel::text::tokenizer::error_code(
                           emel::text::tokenizer::error::model_invalid));

  emel::text::tokenizer::action::finalize(tok_runtime, ctx);
  CHECK(tok_ctx.err ==
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none));
  CHECK(tok_ctx.result);

  tok_ctx.token_count = 2;
  emel::text::tokenizer::action::on_unexpected(tok_runtime, ctx);
  CHECK(tok_ctx.token_count == 0);
  CHECK(tok_ctx.err == emel::text::tokenizer::error_code(
                           emel::text::tokenizer::error::invalid_request));

  emel::text::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = &vocab;
  emel::text::tokenizer::event::bind_ctx bind_ctx = {};
  emel::text::tokenizer::event::bind_runtime bind_runtime{bind_ev, bind_ctx};
  emel::text::tokenizer::action::on_unexpected(bind_runtime, ctx);
  CHECK(bind_ctx.err == emel::text::tokenizer::error_code(
                            emel::text::tokenizer::error::invalid_request));
}
