#include <array>

#include <doctest/doctest.h>

#include "emel/model/data.hpp"
#include "emel/tokenizer/actions.hpp"
#include "emel/tokenizer/guards.hpp"

namespace {

emel::model::data::vocab make_vocab_for_specials() {
  emel::model::data::vocab vocab = {};
  vocab.add_bos = true;
  vocab.add_eos = true;
  vocab.add_sep = true;
  vocab.bos_id = 1;
  vocab.eos_id = 2;
  vocab.sep_id = 3;
  return vocab;
}

}  // namespace

TEST_CASE("tokenizer_detail_model_kind_mappings") {
  using model = emel::model::data::tokenizer_model;
  using encoder_kind = emel::tokenizer::action::encoder_kind;
  using pre_kind = emel::tokenizer::action::preprocessor_kind;

  CHECK(emel::tokenizer::detail::encoder_kind_from_model(model::SPM) ==
        encoder_kind::spm);
  CHECK(emel::tokenizer::detail::encoder_kind_from_model(model::BPE) ==
        encoder_kind::bpe);
  CHECK(emel::tokenizer::detail::encoder_kind_from_model(model::WPM) ==
        encoder_kind::wpm);
  CHECK(emel::tokenizer::detail::encoder_kind_from_model(model::UGM) ==
        encoder_kind::ugm);
  CHECK(emel::tokenizer::detail::encoder_kind_from_model(model::RWKV) ==
        encoder_kind::rwkv);
  CHECK(emel::tokenizer::detail::encoder_kind_from_model(model::PLAMO2) ==
        encoder_kind::plamo2);
  CHECK(emel::tokenizer::detail::encoder_kind_from_model(model::NONE) ==
        encoder_kind::fallback);
  CHECK(emel::tokenizer::detail::encoder_kind_from_model(model::UNKNOWN) ==
        encoder_kind::fallback);

  CHECK(emel::tokenizer::detail::preprocessor_kind_from_model(model::SPM) ==
        pre_kind::spm);
  CHECK(emel::tokenizer::detail::preprocessor_kind_from_model(model::BPE) ==
        pre_kind::bpe);
  CHECK(emel::tokenizer::detail::preprocessor_kind_from_model(model::WPM) ==
        pre_kind::wpm);
  CHECK(emel::tokenizer::detail::preprocessor_kind_from_model(model::UGM) ==
        pre_kind::ugm);
  CHECK(emel::tokenizer::detail::preprocessor_kind_from_model(model::RWKV) ==
        pre_kind::rwkv);
  CHECK(emel::tokenizer::detail::preprocessor_kind_from_model(model::PLAMO2) ==
        pre_kind::plamo2);
  CHECK(emel::tokenizer::detail::preprocessor_kind_from_model(model::NONE) ==
        pre_kind::fallback);
  CHECK(emel::tokenizer::detail::preprocessor_kind_from_model(model::UNKNOWN) ==
        pre_kind::fallback);
}

TEST_CASE("tokenizer_guard_prefix_suffix_cases") {
  auto vocab = make_vocab_for_specials();
  emel::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;
  ctx.add_special = true;
  ctx.token_capacity = 1;
  ctx.token_count = 0;
  ctx.model_kind = emel::tokenizer::action::encoder_kind::bpe;

  CHECK(emel::tokenizer::guard::bos_ready{}(ctx));
  CHECK_FALSE(emel::tokenizer::guard::bos_invalid_id{}(ctx));
  CHECK_FALSE(emel::tokenizer::guard::bos_no_capacity{}(ctx));

  ctx.token_count = ctx.token_capacity;
  CHECK(emel::tokenizer::guard::bos_no_capacity{}(ctx));

  ctx.token_count = 0;
  vocab.bos_id = -1;
  CHECK(emel::tokenizer::guard::bos_invalid_id{}(ctx));

  vocab.bos_id = 1;
  ctx.model_kind = emel::tokenizer::action::encoder_kind::wpm;
  CHECK(emel::tokenizer::guard::sep_ready{}(ctx));
  ctx.token_count = ctx.token_capacity;
  CHECK(emel::tokenizer::guard::sep_no_capacity{}(ctx));

  ctx.token_count = 0;
  vocab.sep_id = -1;
  CHECK(emel::tokenizer::guard::sep_invalid_id{}(ctx));

  vocab.sep_id = 3;
  ctx.model_kind = emel::tokenizer::action::encoder_kind::bpe;
  CHECK(emel::tokenizer::guard::eos_ready{}(ctx));
  ctx.token_count = ctx.token_capacity;
  CHECK(emel::tokenizer::guard::eos_no_capacity{}(ctx));

  ctx.token_count = 0;
  vocab.eos_id = -1;
  CHECK(emel::tokenizer::guard::eos_invalid_id{}(ctx));
}

TEST_CASE("tokenizer_guard_can_tokenize") {
  auto vocab = make_vocab_for_specials();
  emel::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;
  ctx.is_bound = true;

  std::array<int32_t, 2> tokens = {};
  int32_t count = 0;
  emel::tokenizer::event::tokenize ev = {};
  ev.vocab = &vocab;
  ev.token_ids_out = tokens.data();
  ev.token_capacity = static_cast<int32_t>(tokens.size());
  ev.token_count_out = &count;

  CHECK(emel::tokenizer::guard::can_tokenize{}(ev, ctx));

  ev.vocab = nullptr;
  CHECK_FALSE(emel::tokenizer::guard::can_tokenize{}(ev, ctx));

  ev.vocab = &vocab;
  ev.token_ids_out = nullptr;
  CHECK_FALSE(emel::tokenizer::guard::can_tokenize{}(ev, ctx));
}

TEST_CASE("tokenizer_detail_append_token_errors") {
  emel::tokenizer::action::context ctx{};
  std::array<int32_t, 1> tokens = {};
  ctx.token_ids_out = tokens.data();
  ctx.token_capacity = static_cast<int32_t>(tokens.size());
  ctx.token_count = 0;

  CHECK_FALSE(emel::tokenizer::detail::append_token(ctx, -1));
  ctx.token_ids_out = nullptr;
  CHECK_FALSE(emel::tokenizer::detail::append_token(ctx, 1));
  ctx.token_ids_out = tokens.data();
  ctx.token_capacity = 0;
  CHECK_FALSE(emel::tokenizer::detail::append_token(ctx, 1));
}

TEST_CASE("tokenizer_actions_error_paths") {
  auto vocab = make_vocab_for_specials();
  emel::tokenizer::action::context ctx{};
  ctx.vocab = &vocab;

  emel::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = nullptr;
  emel::tokenizer::action::reject_bind(bind_ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.vocab = nullptr;
  emel::tokenizer::action::bind_preprocessor(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.vocab = nullptr;
  emel::tokenizer::action::bind_encoder(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.vocab = nullptr;
  emel::tokenizer::action::run_preprocess(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.vocab = &vocab;
  std::array<int32_t, 1> out_tokens = {};
  ctx.token_ids_out = out_tokens.data();
  ctx.token_capacity = static_cast<int32_t>(out_tokens.size());
  ctx.token_count = 0;
  ctx.last_error = EMEL_OK;
  ctx.phase_error = EMEL_OK;
  emel::tokenizer::action::append_bos(ctx);
  CHECK(ctx.last_error == EMEL_OK);
  CHECK(ctx.token_count == 1);

  ctx.token_count = 0;
  ctx.token_capacity = 0;
  emel::tokenizer::action::append_bos(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.token_ids_out = out_tokens.data();
  ctx.token_capacity = 1;
  vocab.sep_id = -1;
  emel::tokenizer::action::append_sep(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.token_ids_out = nullptr;
  emel::tokenizer::action::append_eos(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.fragment_count = 0;
  ctx.fragment_index = 0;
  emel::tokenizer::action::encode_raw_fragment(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.fragment_count = 1;
  ctx.fragment_index = 0;
  ctx.fragments[0].kind = emel::tokenizer::action::fragment_kind::token;
  emel::tokenizer::action::encode_raw_fragment(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.fragments[0].kind = emel::tokenizer::action::fragment_kind::raw_text;
  emel::tokenizer::action::append_fragment_token(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.fragments[0].kind = emel::tokenizer::action::fragment_kind::raw_text;
  ctx.token_ids_out = nullptr;
  emel::tokenizer::action::encode_raw_fragment(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_guard_fragment_selection") {
  emel::tokenizer::action::context ctx{};
  ctx.fragment_count = 1;
  ctx.fragment_index = 0;
  ctx.fragments[0].kind = emel::tokenizer::action::fragment_kind::token;
  CHECK(emel::tokenizer::guard::more_fragments_token{}(ctx));
  CHECK_FALSE(emel::tokenizer::guard::more_fragments_raw{}(ctx));

  ctx.fragments[0].kind = emel::tokenizer::action::fragment_kind::raw_text;
  CHECK(emel::tokenizer::guard::more_fragments_raw{}(ctx));
}

TEST_CASE("tokenizer_actions_status_helpers") {
  emel::tokenizer::action::context ctx{};
  ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
  emel::tokenizer::action::finalize(ctx);
  CHECK(ctx.last_error == EMEL_OK);

  emel::tokenizer::action::set_capacity_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  emel::tokenizer::action::set_invalid_id_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_MODEL_INVALID);

  emel::tokenizer::event::tokenize ev = {};
  emel::tokenizer::action::on_unexpected(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tokenizer_guard_basic_failures") {
  emel::tokenizer::action::context ctx{};
  emel::tokenizer::event::tokenize ev = {};
  CHECK_FALSE(emel::tokenizer::guard::can_tokenize{}(ev, ctx));

  emel::tokenizer::event::bind bind_ev = {};
  CHECK_FALSE(emel::tokenizer::guard::can_bind{}(bind_ev));

  ctx.add_special = false;
  CHECK(emel::tokenizer::guard::no_prefix{}(ctx));
}
