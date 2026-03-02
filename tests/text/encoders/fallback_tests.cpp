#include "test_support.hpp"

TEST_CASE("encoder_fallback_byte_tokens") {
  vocab_builder builder{};
  builder.set_model("unknown");
  const int32_t x_id = builder.add_byte_token(static_cast<uint8_t>('x'));
  const int32_t y_id = builder.add_byte_token(static_cast<uint8_t>('y'));

  emel::text::encoders::fallback::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "xy",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 2);
  CHECK(tokens[0] == x_id);
  CHECK(tokens[1] == y_id);
}

TEST_CASE("encoder_fallback_encode_requires_prepared_tables") {
  vocab_builder builder{};
  builder.set_model("unknown");
  builder.add_byte_token(static_cast<uint8_t>('x'));

  emel::text::encoders::action::context ctx{};
  ctx.vocab = builder.vocab;

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = "x",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::fallback::detail::encode_fallback(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(result.token_count == 0);
}
