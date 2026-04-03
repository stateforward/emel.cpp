#include "test_support.hpp"

TEST_CASE("encoder_plamo2_byte_tokens") {
  vocab_builder builder{};
  builder.set_model("plamo2");
  builder.add_token("<unk>", 0.0f, 2);
  builder.add_all_plamo2_byte_tokens();
  const int32_t byte_id = builder.add_plamo2_byte_token(static_cast<uint8_t>('p'));

  emel::text::encoders::plamo2::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "p",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(token_count == 1);
  CHECK(tokens[0] == byte_id);
}

TEST_CASE("encoder_detail_plamo2_bom_and_missing_bytes") {
  vocab_builder builder{};
  builder.set_model("plamo2");
  builder.add_token("dummy", 0.0f, 1);
  builder.add_token("", 0.0f, 1);
  builder.add_all_plamo2_byte_tokens();
  emel::text::encoders::plamo2::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::plamo2::detail::ensure_plamo2_tables(ctx, *builder.vocab));
  CHECK(emel::text::encoders::plamo2::detail::ensure_plamo2_tables(ctx, *builder.vocab));
  std::array<int32_t, 8> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::event::encode ev{
    .vocab = *builder.vocab,
    .text = "\xEF\xBB\xBF" "a",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::text::encoders::plamo2::detail::encode_plamo2(ev, ctx, *builder.vocab);
  CHECK(result.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(result.token_count > 0);

  emel::text::encoders::event::encode ev_bom_only = ev;
  ev_bom_only.text = "\xEF\xBB\xBF";
  const auto bom_only =
      emel::text::encoders::plamo2::detail::encode_plamo2(ev_bom_only, ctx, *builder.vocab);
  CHECK(bom_only.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  CHECK(bom_only.token_count == 0);

  emel::text::encoders::event::encode ev_long = ev;
  const size_t max_len = ctx.cpts.size();
  std::string long_text(max_len + 1, 'a');
  ev_long.text = long_text;
  const auto too_long =
      emel::text::encoders::plamo2::detail::encode_plamo2(ev_long, ctx, *builder.vocab);
  CHECK(too_long.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));

  vocab_builder incomplete_builder{};
  incomplete_builder.set_model("plamo2");
  incomplete_builder.add_token("dummy", 0.0f, 1);
  incomplete_builder.add_plamo2_byte_token(static_cast<uint8_t>('a'));
  emel::text::encoders::plamo2::action::context ctx_incomplete{};
  ctx_incomplete.vocab = incomplete_builder.vocab;
  emel::text::encoders::event::encode ev_incomplete = ev;
  ev_incomplete.text = "a";
  const auto invalid =
      emel::text::encoders::plamo2::detail::encode_plamo2(ev_incomplete, ctx_incomplete,
                                                   *incomplete_builder.vocab);
  CHECK(invalid.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::model_invalid));
}
