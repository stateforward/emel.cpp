#include "test_support.hpp"

TEST_CASE("encoder_spm_merges_bigram") {
  vocab_builder builder{};
  builder.set_model("llama");
  const int32_t h_id = builder.add_token("h", 0.1f, 1);
  const int32_t i_id = builder.add_token("i", 0.1f, 1);
  const int32_t hi_id = builder.add_token("hi", 0.9f, 1);
  (void)h_id;
  (void)i_id;

  emel::text::encoders::spm::sm machine{};

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::text::encoders::event::encode{
    .vocab = *builder.vocab,
    .text = "hi",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  }));

  CHECK(err == EMEL_OK);
  CHECK(token_count == 1);
  CHECK(tokens[0] == hi_id);
}

TEST_CASE("encoder_detail_spm_merge_capacity_error") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.set_pre("gpt2");
  builder.add_token("h", 0.1f, 1);
  builder.add_token("i", 0.1f, 1);
  builder.add_token("hi", 0.9f, 1);

  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));

  std::array<int32_t, 1> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = "hi",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(0)),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
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

  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));

  std::array<int32_t, 4> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = "hi",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count >= 1);
}

TEST_CASE("encoder_detail_spm_prefix_after_leading_spaces") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.add_token("\xE2\x96\x81", 0.1f, 1);
  builder.add_all_plamo2_byte_tokens();
  builder.add_token("h", 0.1f, 1);
  builder.add_token("i", 0.1f, 1);
  builder.add_token(" ", 0.1f, 1);
  builder.vocab->add_space_prefix = true;

  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = "  hi",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count >= 1);
}

TEST_CASE("encoder_detail_spm_unescaped_spaces") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.vocab->add_space_prefix = true;
  builder.vocab->escape_whitespaces = false;
  builder.add_token(" ", 0.1f, 1);
  builder.add_token("h", 0.1f, 1);
  builder.add_token("i", 0.1f, 1);

  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = "h i",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count >= 1);
}

TEST_CASE("encoder_detail_spm_suffix_escape_spaces") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.add_token("\xE2\x96\x81", 0.1f, 1);
  builder.add_all_plamo2_byte_tokens();
  builder.add_token("h", 0.1f, 1);
  builder.add_token("i", 0.1f, 1);
  builder.vocab->add_space_prefix = true;
  builder.vocab->treat_whitespace_as_suffix = true;

  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = "hi",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count >= 1);
}

TEST_CASE("encoder_detail_spm_suffix_unescaped_space") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.vocab->add_space_prefix = true;
  builder.vocab->treat_whitespace_as_suffix = true;
  builder.vocab->escape_whitespaces = false;
  builder.add_token(" ", 0.1f, 1);
  builder.add_token("h", 0.1f, 1);
  builder.add_token("i", 0.1f, 1);

  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));

  std::array<int32_t, 8> tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = "hi",
    .token_ids = std::span<int32_t>(tokens.data(), static_cast<size_t>(static_cast<int32_t>(tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count >= 1);
}

TEST_CASE("encoder_detail_spm_prefix_overflow") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.vocab->add_space_prefix = true;
  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));
  const size_t max_bytes = ctx.scratch.buffer.size();
  std::string text(max_bytes, 'a');
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = text,
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_spm_space_overflow") {
  vocab_builder builder{};
  builder.set_model("llama");
  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));
  const size_t max_bytes = ctx.scratch.buffer.size();
  std::string text(max_bytes - 1, 'a');
  text.back() = ' ';
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = text,
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("encoder_detail_spm_missing_byte_token") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.add_token("a", 0.0f, 1);
  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = "b",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };
  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_BACKEND);
}

TEST_CASE("encoder_detail_spm_empty_text") {
  vocab_builder builder{};
  builder.set_model("llama");
  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = "",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_OK);
  CHECK(result.token_count == 0);
}

TEST_CASE("encoder_spm_encode_requires_prepared_tables") {
  vocab_builder builder{};
  builder.set_model("llama");
  builder.add_token("a", 0.1f, 1);

  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  ctx.tables_ready = false;

  std::array<int32_t, 2> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = "a",
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(out_tokens.size())),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(result.token_count == 0);
}

TEST_CASE("encoder_detail_spm_symbol_overflow") {
  vocab_builder builder{};
  builder.set_model("llama");
  emel::text::encoders::spm::action::context ctx{};
  ctx.vocab = builder.vocab;
  CHECK(emel::text::encoders::spm::detail::ensure_spm_tables(ctx));
  const size_t max_symbols = ctx.scratch.offsets.size();
  std::string text(max_symbols + 1, 'a');
  std::array<int32_t, 4> out_tokens = {};
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
  emel::text::encoders::event::encode ev{
    .text = text,
    .token_ids = std::span<int32_t>(out_tokens.data(), static_cast<size_t>(static_cast<int32_t>(out_tokens.size()))),
    .token_count_out = &token_count,
    .error_out = &err,
  };

  const auto result = emel::text::encoders::spm::detail::encode_spm(ev, ctx, *builder.vocab);
  CHECK(result.error == EMEL_ERR_INVALID_ARGUMENT);
}
