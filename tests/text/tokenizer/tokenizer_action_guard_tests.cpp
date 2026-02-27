#include <array>
#include <cstring>

#include <doctest/doctest.h>

#include "emel/model/data.hpp"
#include "emel/text/tokenizer/actions.hpp"
#include "emel/text/tokenizer/guards.hpp"

namespace {

emel::model::data::vocab & make_vocab_for_specials() {
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

}  // namespace

TEST_CASE("tokenizer_guard_can_bind_requires_explicit_valid_variants") {
  auto & vocab = make_vocab_for_specials();
  emel::text::tokenizer::event::bind bind_ev = {};

  CHECK_FALSE(emel::text::tokenizer::guard::can_bind{}(bind_ev));

  bind_ev.vocab = &vocab;
  bind_ev.preprocessor_variant = emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
  bind_ev.encoder_variant = emel::text::encoders::encoder_kind::spm;
  CHECK(emel::text::tokenizer::guard::can_bind{}(bind_ev));

  bind_ev.preprocessor_variant =
      static_cast<emel::text::tokenizer::preprocessor::preprocessor_kind>(255);
  CHECK_FALSE(emel::text::tokenizer::guard::can_bind{}(bind_ev));

  bind_ev.preprocessor_variant = emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
  bind_ev.encoder_variant = static_cast<emel::text::encoders::encoder_kind>(255);
  CHECK_FALSE(emel::text::tokenizer::guard::can_bind{}(bind_ev));
}

TEST_CASE("tokenizer_guard_prefix_suffix_cases") {
  auto & vocab = make_vocab_for_specials();
  emel::text::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;
  ctx.add_special = true;
  ctx.token_capacity = 1;
  ctx.token_count = 0;
  ctx.model_kind = emel::text::tokenizer::action::encoder_kind::bpe;

  CHECK(emel::text::tokenizer::guard::bos_ready{}(ctx));
  CHECK_FALSE(emel::text::tokenizer::guard::bos_invalid_id{}(ctx));
  CHECK_FALSE(emel::text::tokenizer::guard::bos_no_capacity{}(ctx));

  ctx.token_count = ctx.token_capacity;
  CHECK(emel::text::tokenizer::guard::bos_no_capacity{}(ctx));

  ctx.token_count = 0;
  vocab.bos_id = -1;
  CHECK(emel::text::tokenizer::guard::bos_invalid_id{}(ctx));

  vocab.bos_id = 1;
  ctx.model_kind = emel::text::tokenizer::action::encoder_kind::wpm;
  CHECK(emel::text::tokenizer::guard::sep_ready{}(ctx));
  ctx.token_count = ctx.token_capacity;
  CHECK(emel::text::tokenizer::guard::sep_no_capacity{}(ctx));

  ctx.token_count = 0;
  vocab.sep_id = -1;
  CHECK(emel::text::tokenizer::guard::sep_invalid_id{}(ctx));

  vocab.sep_id = 3;
  ctx.model_kind = emel::text::tokenizer::action::encoder_kind::bpe;
  CHECK(emel::text::tokenizer::guard::eos_ready{}(ctx));
  ctx.token_count = ctx.token_capacity;
  CHECK(emel::text::tokenizer::guard::eos_no_capacity{}(ctx));

  ctx.token_count = 0;
  vocab.eos_id = -1;
  CHECK(emel::text::tokenizer::guard::eos_invalid_id{}(ctx));
}

TEST_CASE("tokenizer_guard_can_tokenize") {
  auto & vocab = make_vocab_for_specials();
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

  CHECK(emel::text::tokenizer::guard::can_tokenize{}(ev, ctx));

  ev.vocab = nullptr;
  CHECK_FALSE(emel::text::tokenizer::guard::can_tokenize{}(ev, ctx));

  ev.vocab = &vocab;
  ev.token_ids_out = nullptr;
  CHECK_FALSE(emel::text::tokenizer::guard::can_tokenize{}(ev, ctx));
}

TEST_CASE("tokenizer_detail_append_token_errors") {
  emel::text::tokenizer::action::context ctx{};
  std::array<int32_t, 1> tokens = {};
  ctx.token_ids_out = tokens.data();
  ctx.token_capacity = static_cast<int32_t>(tokens.size());
  ctx.token_count = 0;

  CHECK_FALSE(emel::text::tokenizer::detail::append_token(ctx, -1));
  ctx.token_ids_out = nullptr;
  CHECK_FALSE(emel::text::tokenizer::detail::append_token(ctx, 1));
  ctx.token_ids_out = tokens.data();
  ctx.token_capacity = 0;
  CHECK_FALSE(emel::text::tokenizer::detail::append_token(ctx, 1));
}

TEST_CASE("tokenizer_actions_error_paths") {
  auto & vocab = make_vocab_for_specials();
  emel::text::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;

  emel::text::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = nullptr;
  bind_ev.preprocessor_variant = emel::text::tokenizer::preprocessor::preprocessor_kind::bpe;
  bind_ev.encoder_variant = emel::text::encoders::encoder_kind::bpe;
  emel::text::tokenizer::action::reject_bind(bind_ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.vocab = nullptr;
  emel::text::tokenizer::action::bind_preprocessor(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.vocab = nullptr;
  emel::text::tokenizer::action::bind_encoder(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.vocab = nullptr;
  emel::text::tokenizer::action::run_preprocess(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.vocab = &vocab;
  std::array<int32_t, 1> out_tokens = {};
  ctx.token_ids_out = out_tokens.data();
  ctx.token_capacity = static_cast<int32_t>(out_tokens.size());
  ctx.token_count = 0;
  ctx.last_error = EMEL_OK;
  ctx.phase_error = EMEL_OK;
  emel::text::tokenizer::action::append_bos(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.token_count == 1);

  ctx.token_count = 0;
  ctx.token_capacity = 0;
  emel::text::tokenizer::action::append_bos(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.token_ids_out = out_tokens.data();
  ctx.token_capacity = 1;
  vocab.sep_id = -1;
  emel::text::tokenizer::action::append_sep(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.token_ids_out = nullptr;
  emel::text::tokenizer::action::append_eos(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.fragment_count = 0;
  ctx.fragment_index = 0;
  emel::text::tokenizer::action::encode_raw_fragment(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.fragment_count = 1;
  ctx.fragment_index = 0;
  ctx.fragments[0].kind = emel::text::tokenizer::action::fragment_kind::token;
  emel::text::tokenizer::action::encode_raw_fragment(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.fragments[0].kind = emel::text::tokenizer::action::fragment_kind::raw_text;
  emel::text::tokenizer::action::append_fragment_token(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.fragments[0].kind = emel::text::tokenizer::action::fragment_kind::raw_text;
  ctx.token_ids_out = nullptr;
  emel::text::tokenizer::action::encode_raw_fragment(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_guard_fragment_selection") {
  emel::text::tokenizer::action::context ctx{};
  ctx.fragment_count = 1;
  ctx.fragment_index = 0;
  ctx.fragments[0].kind = emel::text::tokenizer::action::fragment_kind::token;
  CHECK(emel::text::tokenizer::guard::more_fragments_token{}(ctx));
  CHECK_FALSE(emel::text::tokenizer::guard::more_fragments_raw{}(ctx));

  ctx.fragments[0].kind = emel::text::tokenizer::action::fragment_kind::raw_text;
  CHECK(emel::text::tokenizer::guard::more_fragments_raw{}(ctx));
}

TEST_CASE("tokenizer_actions_status_helpers") {
  emel::text::tokenizer::action::context ctx{};
  ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
  emel::text::tokenizer::action::finalize(ctx);
  CHECK(ctx.last_error == EMEL_OK);

  emel::text::tokenizer::action::set_capacity_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  emel::text::tokenizer::action::set_invalid_id_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_MODEL_INVALID);

  emel::text::tokenizer::event::tokenize ev = {};
  emel::text::tokenizer::action::on_unexpected(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_guard_basic_failures") {
  emel::text::tokenizer::action::context ctx{};
  emel::text::tokenizer::event::tokenize ev = {};
  CHECK_FALSE(emel::text::tokenizer::guard::can_tokenize{}(ev, ctx));

  emel::text::tokenizer::event::bind bind_ev = {};
  CHECK_FALSE(emel::text::tokenizer::guard::can_bind{}(bind_ev));

  ctx.add_special = false;
  CHECK(emel::text::tokenizer::guard::no_prefix{}(ctx));
}
