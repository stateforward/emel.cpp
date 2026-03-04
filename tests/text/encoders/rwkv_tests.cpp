#include "test_support.hpp"

TEST_CASE("encoder_rwkv_byte_tokens") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  const int32_t byte_id = builder.add_byte_token(static_cast<uint8_t>('r'));

  emel::text::encoders::rwkv::sm machine{};

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "r",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == byte_id);
}

TEST_CASE("encoder_detail_rwkv_unescape_branches") {
  std::string out;
  CHECK(emel::text::encoders::rwkv::detail::unescape_rwkv_token("plain", out));
  CHECK(out == "plain");
  CHECK(emel::text::encoders::rwkv::detail::unescape_rwkv_token("\\n\\t\\r", out));
  CHECK(out == std::string("\n\t\r"));
  CHECK(emel::text::encoders::rwkv::detail::unescape_rwkv_token("\\\\", out));
  CHECK(out == "\\");
  CHECK(emel::text::encoders::rwkv::detail::unescape_rwkv_token("\\x41\\x42", out));
  CHECK(out == "AB");
  CHECK_FALSE(emel::text::encoders::rwkv::detail::unescape_rwkv_token("\\x1", out));
}

TEST_CASE("encoder_rwkv_tables_reject_incomplete_hex") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  builder.add_token("\\x1", 0.0f, 1);
  emel::text::encoders::rwkv::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK_FALSE(emel::text::encoders::rwkv::detail::ensure_rwkv_tables(ctx, *builder.vocab));
}

TEST_CASE("encoder_rwkv_skips_unknown_without_unk") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  builder.add_token("a", 0.0f, 1);
  builder.vocab->unk_id = emel::text::encoders::detail::k_token_null;

  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::rwkv::sm machine{};
  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "b",
    .token_ids = std::span<int32_t>(
      out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(token_count == 0);
}

TEST_CASE("encoder_rwkv_table_cache_and_empty_token") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  builder.add_token("", 0.0f, 1);
  builder.add_token("a", 0.0f, 1);
  emel::text::encoders::rwkv::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::rwkv::detail::ensure_rwkv_tables(ctx, *builder.vocab));
  CHECK(emel::text::encoders::rwkv::detail::ensure_rwkv_tables(ctx, *builder.vocab));
}

TEST_CASE("encoder_rwkv_encode_reports_invalid_table") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  builder.add_token("\\x1", 0.0f, 1);
  std::array<int32_t, 2> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::rwkv::sm machine{};
  CHECK_FALSE(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "a",
    .token_ids = std::span<int32_t>(
      out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_rwkv_push_unk_overflow") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  const int32_t unk_id = builder.add_token("<unk>", 0.0f, 1);
  builder.vocab->unk_id = unk_id;
  std::array<int32_t, 1> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::rwkv::sm machine{};
  CHECK_FALSE(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "z",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(0)),
    .token_count_out = &token_count,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_rwkv_encode_builds_tables_when_missing") {
  vocab_builder builder{};
  builder.set_model("rwkv");
  const int32_t token_id = builder.add_byte_token(static_cast<uint8_t>('a'));

  std::array<int32_t, 2> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::rwkv::sm machine{};
  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "a",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(out_tokens.size())),
    .token_count_out = &token_count,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(out_tokens[0] == token_id);
}
